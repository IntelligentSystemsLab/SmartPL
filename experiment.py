"""
This file contains a class of Experiment, which init env and model etc, and control the process of model training or testing.

"""
import os
from functools import partial

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, vec_monitor
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy, MaskableActorCriticPolicy, MaskableActorCriticCnnPolicy
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList

from callbacks import SaveOnBestTrainingRewardCallback, HParamsCallback
from custom_model.custom_feature_extractor import CustomGCN, CustomCNN, CustomGAT, CustomGraphSAGE
from custom_model.custom_feature_extractor import A2CGraphPolicy


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()


def reset_filename(base_path, base_name, start_num=0):
    """
    Find the next available filename with an incremented numeric suffix.
    
    Args:
        base_path (str): The path where the file should be saved.
        base_name (str): The base name of the file, without the numeric suffix.
        start_num (int): The initial number to start checking from (default: 0).

    Returns:
        str: The next available filename, including the path and the incremented suffix.
    """
    num = start_num
    while True:
        filename = f"{base_name}_{num}"
        full_path = os.path.join(base_path, filename)

        if not os.path.exists(full_path):
            return full_path

        num += 1


class Experiment:

    def __init__(self,
                 EnvClass,
                 ModelClass,
                 config,
                 curriculum_learning=True,
                 load_path='',
                 debug=True):
        """Instantiate the Experiment class.

        Parameters
        ----------
        flow_params : dict
        """
        log_dir = config['log_dir']
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.config = config
        monitor_log_dir = os.path.join(log_dir, 'Monitor')
        monitor_log = reset_filename(monitor_log_dir, 'monitor')
        os.makedirs(monitor_log, exist_ok=True)
        save_best_callback = SaveOnBestTrainingRewardCallback(
            check_freq=500,
            log_dir=monitor_log,
            env_name=EnvClass.__name__,
            config=config)
        save_hyperparam_callback = HParamsCallback(config,
                                                   env_name=EnvClass.__name__)
        Callbacks = CallbackList(
            [save_best_callback, save_hyperparam_callback])

        # setting
        self.total_timesteps = int(float(config['total_timesteps']))
        self.Callbacks = Callbacks

        print("\n============Init SUMO Simulation===========\n")
        maskable = ModelClass.__name__ == 'MaskablePPO' or ModelClass.__name__ == 'Graph_MaskablePPO'
        # self.test_env = EnvClass(None,
        #                          config['Envs'][EnvClass.__name__],
        #                          label='Test')
        self.test_env = EnvClass('human',
                                 config['Envs'][EnvClass.__name__],
                                 label='Test')
        if debug:
            self.env = EnvClass("human", config['Envs'][EnvClass.__name__])
            tensorboard_log = None
        else:
            self.env = EnvClass(None, config['Envs'][EnvClass.__name__])
            # self.env = EnvClass("human", config['Envs'][EnvClass.__name__])
            tensorboard_log = os.path.join(log_dir, ModelClass.__name__,
                                           EnvClass.__name__)
        self.env = Monitor(self.env, filename=monitor_log)
        if maskable:
            self.env = ActionMasker(self.env, mask_fn)
            self.test_env = ActionMasker(self.test_env, mask_fn)
        self.env = DummyVecEnv([lambda: self.env])
        self.env = VecNormalize(self.env, norm_reward=True, norm_obs=False)

        print("\n============Init Agent===========\n")
        if maskable:
            if 'Graph' in EnvClass.__name__:
                model_type = {
                    'GCN': CustomGCN,
                    'GAT': CustomGAT,
                    'GraphSAGE': CustomGraphSAGE,
                }
                policy = MaskableMultiInputActorCriticPolicy
                extractor_type = config['extractor']['type']
                model_class = model_type[extractor_type]
                del config['extractor']['type']
                policy_kwargs = dict(
                    features_extractor_class=model_class,
                    features_extractor_kwargs=dict(**config['extractor']))
                config['extractor']['type'] = extractor_type
            elif 'Grid' in EnvClass.__name__:
                policy = MaskableActorCriticPolicy
                policy_kwargs = dict(
                    features_extractor_class=CustomCNN,
                    features_extractor_kwargs=dict(features_dim=128),
                )
            else:
                policy = MaskableActorCriticPolicy
                policy_kwargs = None
        else:
            if 'Graph' in EnvClass.__name__:
                policy = 'MultiInputPolicy'
                # policy = A2CGraphPolicy
                model_type = {
                    'GCN': CustomGCN,
                    'GAT': CustomGAT,
                    'GraphSAGE': CustomGraphSAGE,
                }
                extractor_type = config['extractor']['type']
                model_class = model_type[extractor_type]
                del config['extractor']['type']
                policy_kwargs = dict(
                    features_extractor_class=model_class,
                    features_extractor_kwargs=dict(**config['extractor']))
                config['extractor']['type'] = extractor_type
            elif 'Grid' in EnvClass.__name__:
                policy = 'MlpPolicy'
                policy_kwargs = dict(
                    features_extractor_class=CustomCNN,
                    features_extractor_kwargs=dict(features_dim=128),
                )
            else:
                policy = 'MlpPolicy'
                policy_kwargs = None
                # policy_kwargs = dict(net_arch=config['Models'][ModelClass.__name__]['net_arch'])
                # del config['Models'][ModelClass.__name__]['net_arch']

        self.model_class = ModelClass
        
        self.model = ModelClass(policy,
                                self.env,
                                policy_kwargs=policy_kwargs,
                                **config['Models'][ModelClass.__name__],
                                verbose=1,
                                tensorboard_log=tensorboard_log)
        if curriculum_learning:
            print('\nusing curriculum learning and loading model\n')
            self.model.load(load_path, env=self.env)

    def train(self):

        self.model.learn(
            total_timesteps=self.total_timesteps,
            progress_bar=True,
            callback=self.Callbacks,
        )
        self.env.close()

    def test(self, env_name, model_path='', round=10):
        if model_path == '':
            log_name = os.path.split(self.model.logger.dir)[-1]
            model_path = os.path.join(self.config['save_dir'],
                                      self.model.__class__.__name__, env_name,
                                      log_name, 'best_model.zip')

        print("\n============Loading {} {} Model===========\n".format(
            self.model.__class__.__name__, env_name))
        test_model = self.model.load(model_path, env=self.test_env)
        print("\n============Loading {} {} Model===========\n".format(
            self.model.__class__.__name__, env_name))

        rewards = []
        fuels = []
        sim_times = []
        for i in range(round):
            reward = []
            sim_time =0
            fuel = np.zeros((1, 4))
            obs, info = self.test_env.reset()
            while True:
                action, _ = test_model.predict(observation=obs,
                                               deterministic=True)

                obs, r, terminated, truncated, info = self.test_env.step(
                    action)
                reward.append(r)
                fuel += info['fuel consumption']

                if terminated:
                    # if info['crash']:
                    #     crash_count += 1
                    # else:
                    sim_time += info['simulation step']
                    sim_times.append(sim_time)
                    break
                if terminated or truncated:
                    if not info['crash']:
                        fuels.append(fuel)
                    break
            rewards.append(reward)

        mean_reward = np.mean(rewards)
        # fuel comsumption
        mean_fuel = np.mean(fuel)
        # time loss
        mean_time = np.mean(sim_times)

        #collision rate
        std_reward = np.std(rewards)
        self.test_env.close()
        return mean_reward, std_reward, mean_fuel,mean_time
