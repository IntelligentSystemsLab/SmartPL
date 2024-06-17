import os

import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self,
                 check_freq: int,
                 config,
                 log_dir,
                 env_name,
                 verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.config = config
        self.save_dir = self.config['save_dir']
        self.env_name = env_name
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        log_name = os.path.split(self.model.logger.dir)[-1]
        self.save_best_path = os.path.join(self.save_dir,
                                           self.model.__class__.__name__,
                                           self.env_name, log_name,
                                           'best_model')

        # if self.save_best_path is not None:
        #     os.makedirs(self.save_best_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(
                            f"Saving new best model to {self.save_best_path}")
                    self.model.save(self.save_best_path)

        return True


class HParamsCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def __init__(self, config, env_name, verbose: int = 1):
        super().__init__(verbose)
        self.config = config
        self.save_dir = self.config['save_dir']
        self.env_name = env_name

    def _on_training_start(self) -> None:
        mean_reward, std_reward = evaluate_policy(self.model,
                                                  self.training_env,
                                                  n_eval_episodes=5,
                                                  use_masking=False,
                                                  deterministic=False)
        print("\n============Random Action===========\n")
        print('mean_reward:', mean_reward)
        print('std_reward:', std_reward)
        print("\n============Random Action===========\n")

        self.hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            # 'type':
            # self.model.policy_kwargs['features_extractor_class'].__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "total_timesteps": self.locals['total_timesteps'],
            **self.config['extractor'],
            # 'n_steps': self.model.n_steps,
            # 'n_epochs': self.model.n_epochs
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            'test_mean_reward': 0.0,
            'test_std_reward': 0.0,
        }

        # for initialize the scaler for hparam log
        self.logger.record("test_mean_reward", 10)
        self.logger.record("test_std_reward", 10)
        self.logger.dump(self.num_timesteps)

        self.logger.record(
            "hparams",
            HParam(self.hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:

        log_name = os.path.split(self.model.logger.dir)[-1]
        self.save_latest_path = os.path.join(self.save_dir,
                                             self.model.__class__.__name__,
                                             self.env_name, log_name,
                                             'latest_model')
        self.save_best_path = os.path.join(self.save_dir,
                                           self.model.__class__.__name__,
                                           self.env_name, log_name,
                                           'best_model.zip')
        self.model.save(self.save_latest_path)

        print('\n Loading Best Model.\n')
        best_model = self.model.load(self.save_best_path)

        mean_reward, std_reward = evaluate_policy(best_model,
                                                  self.training_env,
                                                  use_masking=False,
                                                  n_eval_episodes=30,
                                                  deterministic=True)
        print("\n============Trained Agent===========\n")
        print('mean_reward:', mean_reward)
        print('std_reward:', std_reward)
        print("\n============Trained Agent===========\n")
        self.logger.record("test_mean_reward", mean_reward)
        self.logger.record("test_std_reward", std_reward)

        self.logger.dump(self.num_timesteps)
        return True

