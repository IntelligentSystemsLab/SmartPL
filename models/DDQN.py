from torch.nn import functional as F
import torch as th
from stable_baselines3 import DQN
import numpy as np


class DDQN(DQN):

    def __init__(self,
                 policy,
                 env,
                 learning_rate: float = 0.0001,
                 buffer_size: int = 50000,
                 learning_starts: int = 500,
                 batch_size: int = 32,
                 tau: float = 1,
                 gamma: float = 0.99,
                 train_freq: int = 4,
                 gradient_steps: int = 1,
                 replay_buffer_class: None = None,
                 replay_buffer_kwargs: None = None,
                 optimize_memory_usage: bool = False,
                 target_update_interval: int = 10000,
                 exploration_fraction: float = 0.1,
                 exploration_initial_eps: float = 1,
                 exploration_final_eps: float = 0.05,
                 max_grad_norm: float = 10,
                 stats_window_size: int = 100,
                 tensorboard_log: None = None,
                 policy_kwargs: None = None,
                 verbose: int = 0,
                 seed: None = None,
                 device: str = "auto",
                 _init_setup_model: bool = True) -> None:
        super().__init__(policy, env, learning_rate, buffer_size,
                         learning_starts, batch_size, tau, gamma, train_freq,
                         gradient_steps, replay_buffer_class,
                         replay_buffer_kwargs, optimize_memory_usage,
                         target_update_interval, exploration_fraction,
                         exploration_initial_eps, exploration_final_eps,
                         max_grad_norm, stats_window_size, tensorboard_log,
                         policy_kwargs, verbose, seed, device,
                         _init_setup_model)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            ### YOUR CODE HERE
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            # Do not backpropagate gradient to the target network
            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(
                    replay_data.next_observations).max(dim=1)[0].unsqueeze(-1)
                # Compute q-values for the next observation using the online q net
                next_q_values_online = self.q_net(
                    replay_data.next_observations).max(dim=1)[0].unsqueeze(-1)
                # Select action with online network
                next_actions_online = next_q_values_online.argmax(dim=1, keepdim=True)
                # Estimate the q-values for the selected actions using target q network
                next_q_values_target = next_q_values.gather(dim=1, index=next_actions_online)

                # 1-step TD target
                target_q_values = replay_data.rewards + (
                    1 - replay_data.dones) * self.gamma * next_q_values_target

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values,
                                         dim=1,
                                         index=replay_data.actions)

            # Compute loss (Huber loss)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            losses.append(loss.item())

            # Optimize the q-network
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                        self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates",
                           self._n_updates,
                           exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
