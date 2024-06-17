# Comparison with baseline model in different situation ,sush as hdv inverval, lane count
import os

import yaml
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd

import envs
from envs import PLDriving_highway_v2_CoOP, PLDriving_highway_v2_RuleBased
import models
from custom_model.custom_feature_extractor import CustomGAT

# env_setting
LANE_COUNT = [2,3,4]
HDV_INTERVAL = [2,3,4]
SEED =[0,2001,2023]
# SEED = ['None']
EVAL_ROUND = 1

# MODEL_PATH = r'checkpoints\MaskablePPO\PLDriving_highway_v2_Kinematic\PPO_1\best_model.zip'

# models
# MODELS = ['SmartPL','CoOP','Plexe']
MODELS = ['SmartPL']
# 加载 XML 文件
tree = ET.parse('envs/cfg/freeway.sumo.cfg')
root = tree.getroot()

# # 找到 <input> 元素
input_element = root.find('input')

# # 修改 <net-file> 的值
net_file_element = input_element.find('net-file')

with open('./config.yaml', 'r', encoding='utf-8') as config_file:
    config = yaml.safe_load(config_file)
for index,seed in enumerate(SEED):
    EnvClass = getattr(envs, 'PLDriving_highway_v2_Graph')
    config['Envs']['PLDriving_highway_v2_Graph'].update({'seed': seed})

    Mean_rewards = []
    Mean_fuels = []
    Std_rewards = []
    Mean_sim_times = []
    Crash_rates = []
    Lane_counts = []
    Hdv_intervals = []
    Model_types = []
    for lane_count in LANE_COUNT:
        net_file_element.set('value', '{}_lane_freeway.net.xml'.format(lane_count))
        # 保存修改后的 XML 文件
        tree.write('envs/cfg/freeway.sumo.cfg')
        config['Envs']['PLDriving_highway_v2_Graph'].update(
            {'highway_lanes': lane_count})
        for hdv_interval in HDV_INTERVAL:
            config['Envs']['PLDriving_highway_v2_Graph'].update(
                {'hdv_interval': hdv_interval})
            # env = EnvClass('human',
            #             config['Envs']['PLDriving_highway_v2_Graph'],
            #             label='Test')
            env = EnvClass(None,
                        config['Envs']['PLDriving_highway_v2_Graph'],
                        label='Test')
            for model_type in MODELS:
                model = None
                if model_type == 'SmartPL':
                    ModelClass = getattr(models, 'Graph_MaskablePPO')
                    policy = 'MultiInputPolicy'
                    extractor_type = config['extractor']['type']
                    model_class = CustomGAT
                    try:
                        del config['extractor']['type']
                    except:
                        pass
                    policy_kwargs = dict(
                        features_extractor_class=model_class,
                        features_extractor_kwargs=dict(**config['extractor']))
                    config['extractor']['type'] = extractor_type

                    model = ModelClass(policy,
                                    env,
                                    policy_kwargs=policy_kwargs,
                                    **config['Models'][ModelClass.__name__],
                                    verbose=1)
                    print("\n============Loading {} {} Model===========\n".format(
                        model.__class__.__name__, 'PLDriving_highway_v2_Graph'))
                    # model_path = os.path.join('checkpoints/MaskablePPO/PLDriving_highway_v2_Kinematic',CHECKPOINT[index],'best_model.zip')
                    model_path = os.path.join('checkpoints/Graph_MaskablePPO/PLDriving_highway_v2_Graph','PPO_13','best_model.zip')
                    model = model.load(model_path, env=env)
                    print("\n============Loading {} {} Model===========\n".format(
                        model.__class__.__name__, 'PLDriving_highway_v2_Graph'))
                elif model_type == 'CoOP':
                    config['Envs']['PLDriving_highway_v2_CoOP'].update({'seed': seed})
                    config['Envs']['PLDriving_highway_v2_CoOP'].update(
                        {'highway_lanes': lane_count})
                    # env = PLDriving_highway_v2_CoOP(
                    #     'human', config=config['Envs']['PLDriving_highway_v2_CoOP'])
                    env = PLDriving_highway_v2_CoOP(
                        None, config=config['Envs']['PLDriving_highway_v2_CoOP'])
                else:
                    config['Envs']['PLDriving_highway_v2_RuleBased'].update(
                        {'seed': seed})
                    config['Envs']['PLDriving_highway_v2_RuleBased'].update(
                        {'highway_lanes': lane_count})
                    # env = PLDriving_highway_v2_RuleBased(
                    #     'human',
                    #     config=config['Envs']['PLDriving_highway_v2_RuleBased'])
                    env = PLDriving_highway_v2_RuleBased(
                        None, config=config['Envs']['PLDriving_highway_v2_RuleBased'])

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
                Model_types.append(model_type)

    # 创建字典以存储指标
    data = {
        'mean_reward': Mean_rewards,
        'std_reward': Std_rewards,
        'crash_count': Crash_rates,
        'mean_sim_time': Mean_sim_times,
        'Model_types': Model_types,
        'Hdv_intervals': Hdv_intervals,
        'Lane_counts': Lane_counts,
    }

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 保存为 CSV 文件
    df.to_csv('data/baselines/baselines_{1}.csv'.format(HDV_INTERVAL[0],seed), index=False)
    # df.to_csv('data/EI/baselines_{0}_{1}.csv'.format(HDV_INTERVAL[0],seed), index=False)