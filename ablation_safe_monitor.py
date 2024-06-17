# Comparison with baseline model in different situation ,sush as hdv inverval, lane count
import os

import yaml
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy, MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym

import envs
import models
from custom_model.custom_feature_extractor import CustomGAT,CustomCNN

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()

# env_setting
LANE_COUNT = [4]
HDV_INTERVAL = [2]
# SEED =[0,2001,2023]
SEED = 'None'
EVAL_ROUND = 10

RENDER = None
# RENDER = 'human'

SAFE_MONITOR= [True,False]

# 加载 XML 文件
tree = ET.parse('envs/cfg/freeway.sumo.cfg')
root = tree.getroot()

# # 找到 <input> 元素
input_element = root.find('input')
# # 修改 <net-file> 的值
net_file_element = input_element.find('net-file')

with open('./config.yaml', 'r', encoding='utf-8') as config_file:
    config = yaml.safe_load(config_file)
  
Mean_rewards = []
Mean_fuels = []
Std_rewards = []
Mean_sim_times = []
Crash_rates = []
Lane_counts = []
Hdv_intervals = []
Safe_monitors = []

for lane_count in LANE_COUNT: 
    for hdv_interval in HDV_INTERVAL:
        for safe_monitor in SAFE_MONITOR:
            env_name = 'PLDriving_highway_v2_Graph'
            EnvClass = getattr(envs, env_name)
            config['Envs'][env_name].update({'seed': SEED})
            
            config['Envs'][env_name].update({'safe_monitor': safe_monitor})

            net_file_element.set('value', '{}_lane_freeway.net.xml'.format(lane_count))
            # 保存修改后的 XML 文件
            tree.write('envs/cfg/freeway.sumo.cfg')
            config['Envs'][env_name].update(
                {'highway_lanes': lane_count})       
            
            # hdv interval setting
            config['Envs'][env_name].update(
                    {'hdv_interval': hdv_interval})

            env = EnvClass(RENDER,
                        config['Envs'][env_name],
                        label='Test')
            env = ActionMasker(env, mask_fn)   
            ModelClass = getattr(models, 'MaskablePPO')
            policy = MaskableMultiInputActorCriticPolicy
            policy_kwargs = dict(
                features_extractor_class=CustomGAT,
                features_extractor_kwargs=None)
            model = ModelClass(policy,
                            env,
                            policy_kwargs=policy_kwargs,
                            **config['Models'][ModelClass.__name__],
                            verbose=1)
            print("\n============Loading {} {} Model===========\n".format(
                model.__class__.__name__, env_name))
            model_path = os.path.join(r'checkpoints\Graph_MaskablePPO\PLDriving_highway_v2_Graph\PPO_6\best_model.zip')
            model = model.load(model_path, env=env)
            print("\n============Loading {} {} Model===========\n".format(
                model.__class__.__name__, env_name))
                                    
            rewards = []
            crash_counts = []
            sim_times = []

            for i in range(EVAL_ROUND):

                crash_count = 0
                sim_time = 0
                reward = 0
                fuel = np.zeros((1, 4))

                obs, _ = env.reset()
                while True:
                    if model is not None:
                        action, _ = model.predict(observation=obs,
                                                deterministic=True)
                    else:
                        action = None
                    obs, r, terminated, _, info = env.step(action=action)

                    reward += r
                    if terminated:
                        if info['crash']:
                            crash_count += 1
                        else:
                            sim_time += info['simulation step']
                            sim_times.append(sim_time)
                        break

                # sim_times.append(sim_time)
                crash_counts.append(crash_count)
                rewards.append(reward)
            env.close()

            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            mean_sim_time = np.mean(sim_times)
            crash_rate = sum(crash_counts) / EVAL_ROUND

            Mean_rewards.append(mean_reward)
            Std_rewards.append(std_reward)
            Crash_rates.append(crash_rate)
            Mean_sim_times.append(mean_sim_time)
            Lane_counts.append(lane_count)
            Hdv_intervals.append(hdv_interval)
            Safe_monitors.append(safe_monitor)
        

# 创建字典以存储指标
data = {
    'mean_reward': Mean_rewards,
    'std_reward': Std_rewards,
    'crash_count': Crash_rates,
    'mean_sim_time': Mean_sim_times,
    'safe_monitor': Safe_monitors,
    'Hdv_intervals': Hdv_intervals,
    'Lane_counts': Lane_counts,
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV 文件
df.to_csv('data/safe monitor/interval{0}/safe_monitor_{1}.csv'.format(HDV_INTERVAL[0],SEED), index=False)
