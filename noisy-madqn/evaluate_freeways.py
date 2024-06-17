# evalute NoisyNet-MADQN in various traffic conditions
# Comparison with baseline model in different situation ,sush as hdv inverval, lane count\
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd

from MADQN import MADQN
from env import SumoPettingZooEnv

# model setting
MODEL_TYPE = 'NoisyNet-MADQN'
MAX_STEPS = 1e5
EPISODES_BEFORE_TRAIN = 5000
EVAL_EPISODES = 3
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
SEED =[0,2001,2023]
# SEED = ['None']
EVAL_ROUND = 2


tree = ET.parse('envs/cfg/freeway.sumo.cfg')
root = tree.getroot()


input_element = root.find('input')


net_file_element = input_element.find('net-file')


for index,seed in enumerate(SEED):

    env = SumoPettingZooEnv(render_mode=None,seed=seed)

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
    
    madqn.load('noisy-madqn/checkpoints/seed_None/', 21000)

    Mean_rewards = []
    Lane_counts = []
    Hdv_intervals = []
    Model_types = []
    for lane_count in LANE_COUNT:
        net_file_element.set('value', '{}_lane_freeway.net.xml'.format(lane_count))

        tree.write('envs/cfg/freeway.sumo.cfg')

        env.highway_lanes = lane_count
        for hdv_interval in HDV_INTERVAL:
            env.hdv_interval = hdv_interval
            rewards, _ ,reward_metrics= madqn.evaluation(env, EVAL_EPISODES)

            mean_reward = np.sum(reward_metrics)/ EVAL_EPISODES

            Mean_rewards.append(mean_reward)
            Lane_counts.append(lane_count)
            Hdv_intervals.append(hdv_interval)
            Model_types.append(MODEL_TYPE)

    data = {
        'mean_reward': Mean_rewards,
        'Hdv_intervals': Hdv_intervals,
        'Lane_counts': Lane_counts,
        'Model_types': Model_types,
    }
    df = pd.DataFrame(data)
    env.close()
    df.to_csv('noisy-madqn/results/NoisyNet-MADQN_{0}.csv'.format(seed), index=False)