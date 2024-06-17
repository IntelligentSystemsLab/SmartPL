# 汽车编队的主函数，调用交通仿真、强化学习算法等
import yaml

import envs
import models
from experiment import Experiment

TRAIN = True
# TRAIN = False

with open('./config.yaml', 'r', encoding='utf-8') as config_file:
    config = yaml.safe_load(config_file)

# set seed
for env_name in config['Env']:
    for model_name in config['Model']:
        for seed in config['seed']:
            EnvClass = getattr(envs, env_name)
            # update config 
            config['Envs'][EnvClass.__name__].update({'seed': seed})

            ModelClass = getattr(models, model_name)
            config['total_timesteps'] = 1e5
            load_path = r'checkpoints\Graph_MaskablePPO\PLDriving_highway_v2_Graph\PPO_6\best_model.zip'

            # set up an experiment
            experiment = Experiment(EnvClass,
                                    ModelClass,
                                    config,
                                    curriculum_learning=False,
                                    load_path=load_path,
                                    debug=True
                                    )

            if TRAIN:
                experiment.train()
            else:
                mean_reward, std_reward, fuel_consumption,mean_time = experiment.test(
                    model_path=
                    r'checkpoints\Graph_MaskablePPO\PLDriving_highway_v2_Graph\PPO_13\best_model.zip',
                    round=10,
                    env_name=env_name)

                print("\n============Trained Agent===========\n")
                print('mean_reward:', mean_reward)
                print('std_reward:', std_reward)
                print('fuel_consumption:', fuel_consumption)
                print('mean_time:',mean_time)
                print("\n============Trained Agent===========\n")
