# evalute model during training process.

# evalute NoisyNet-MADQN in various traffic conditions
# Comparison with baseline model in different situation ,sush as hdv inverval, lane count\
import re
import os

import numpy as np
import pandas as pd

from MADQN import MADQN
from env import SumoPettingZooEnv
from single_agent.utils_common import agg_double_list

# model setting
MAX_STEPS = 1e5
EPISODES_BEFORE_TRAIN = 5000
EVAL_EPISODES = 10
EVAL_INTERVAL = 3000

MEMORY_CAPACITY = 1000000
BATCH_SIZE = 64
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

REWARD_DISCOUNTED_GAMMA = 0.99
EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 20000

# env_setting
LANE_COUNT = [2,3,4]
HDV_INTERVAL = [2,3,4]
SEED =None
# SEED = ['None']
EVAL_ROUND = 2



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


mean_rewards = []


checkpoint_dir = 'noisy-madqn/checkpoints/seed_None/'
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]


pattern = re.compile(r'checkpoint-(\d+).pt')


checkpoint_files_with_steps = []
for file in checkpoint_files:
    match = pattern.match(file)
    if match:
        step = int(match.group(1))
        checkpoint_files_with_steps.append((step, file))

checkpoint_files_with_steps.sort()  

for step, file in checkpoint_files_with_steps:
    # checkpoint_path = os.path.join(checkpoint_dir, file)
    

    madqn.load(checkpoint_dir, step)
    

    env.highway_lanes = 4
    env.hdv_interval = 2
    rewards, _, reward_metrics = madqn.evaluation(env, EVAL_EPISODES)
    # rewards_mu, rewards_std = agg_double_list(rewards)
    mean_reward = np.mean([np.sum(np.array(l_i), 0) for l_i in reward_metrics])
    # mean_reward = np.sum(reward_metrics) / EVAL_EPISODES
    mean_rewards.append([mean_reward, step])
    # mean_rewards.append([rewards_mu,step])


mean_rewards_df = pd.DataFrame(mean_rewards, columns=['Reward', 'Step'])
mean_rewards_df.to_csv(os.path.join(checkpoint_dir, 'reward.csv'), index=False)