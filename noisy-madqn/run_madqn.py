import pandas as pd

from MADQN import MADQN
from single_agent.utils_common import agg_double_list
from env import SumoPettingZooEnv


import warnings

warnings.filterwarnings("ignore")

import sys
# sys.path.append("../highway-env")
import gym
import numpy as np
import matplotlib.pyplot as plt

MAX_STEPS = 1e5
EPISODES_BEFORE_TRAIN = 5000
EVAL_EPISODES = 3
EVAL_INTERVAL = 3000
SEED = None

MEMORY_CAPACITY = 1000000
BATCH_SIZE = 64
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

REWARD_DISCOUNTED_GAMMA = 0.99
EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 20000

TRAIN = True
TRAIN_CHECKPOINT = 15000
# TRAIN_CHECKPOINT = None

# TRAIN = False

from platoon_highway_env import PLHighwayEnv
import gymnasium as gym

gym.register(
    id='pl_highway-v0',
    entry_point=PLHighwayEnv,
)


def run():

    # env = gym.make('pl_highway-v0', render_mode='human')

    # env = SumoPettingZooEnv(render_mode="human",seed=SEED)
    env = SumoPettingZooEnv(render_mode=None,seed=SEED)

    state_dim = env.observation_space().shape[1]
    action_dim = env.action_space().n
    madqn = MADQN(env=env,
                  memory_capacity=MEMORY_CAPACITY,
                  state_dim=state_dim,
                  action_dim=action_dim,
                  batch_size=BATCH_SIZE,
                  max_steps=MAX_STEPS,
                  reward_gamma=REWARD_DISCOUNTED_GAMMA,
                  epsilon_start=EPSILON_START,
                  epsilon_end=EPSILON_END,
                  epsilon_decay=EPSILON_DECAY,
                  max_grad_norm=MAX_GRAD_NORM,
                  episodes_before_train=EPISODES_BEFORE_TRAIN)
    if TRAIN:
        if TRAIN_CHECKPOINT != None:
            madqn.load('noisy-madqn/checkpoints/', TRAIN_CHECKPOINT)
            madqn.n_steps = TRAIN_CHECKPOINT
        # episodes = []
        # eval_rewards = []
        steps = []
        reward_metrics = []
        while madqn.n_steps < MAX_STEPS:
            madqn.interact()
            if madqn.n_steps >= EPISODES_BEFORE_TRAIN:
                madqn.train()
            # if madqn.episode_done and ((madqn.n_episodes) % EVAL_INTERVAL
            #                            == 0):
            if ((madqn.n_steps) % EVAL_INTERVAL
                            == 0):
                madqn.save('noisy-madqn/checkpoints/seed_{0}/'.format(SEED), madqn.n_steps)
                rewards, _,reward_metric = madqn.evaluation(env, EVAL_EPISODES)
                reward_metric = np.mean([np.sum(np.array(l_i), 0) for l_i in reward_metric])
                # rewards_mu, rewards_std = agg_double_list(rewards)
                print("Episode %d, Average Reward %.2f" %
                      (madqn.n_steps, reward_metric))
                # episodes.append(madqn.n_steps + 1)
                # eval_rewards.append(rewards_mu)
                reward_metrics.append(reward_metric)
                steps.append(madqn.n_steps)

        # episodes = np.array(episodes)
        # eval_rewards = np.array(eval_rewards)
        reward_metrics = np.array(reward_metrics)
        reward_metrics = pd.DataFrame(reward_metrics,columns=['Reward'])
        reward_metrics.to_csv('noisy-madqn/checkpoints/training_reward.csv', index=False)
        # plt.figure()
        # plt.plot(episodes, eval_rewards)
        # plt.xlabel("Episode")
        # plt.ylabel("Average Reward")
        # plt.legend(["Noise-DQN"])
        # plt.savefig('./results/madqn_seed_None.png', dpi=300)
        # plt.show()

    else:
        madqn.load('noisy-madqn/checkpoints/seed_None/', 69000)
        rewards, _ ,reward_metrics= madqn.evaluation(env, EVAL_EPISODES)
        rewards_mu, rewards_std = agg_double_list(rewards)
        print("Average Reward %.2f" % (rewards_mu))


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run(sys.argv[1])
    else:
        run()
