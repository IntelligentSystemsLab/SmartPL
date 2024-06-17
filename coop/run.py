"""This file runs the CoOP algorithm
    """
from src.c2x import C2X
from src.platoon import Platoon
from envs.highway_gym import PLDriving_highway_v1_Kinematic
import numpy as np
import sumolib
import traci
from plexe import Plexe
import yaml


class PLDriving_highway_v1_COOP(PLDriving_highway_v1_Kinematic):

    def __init__(self, render_mode, config, label=None) -> None:
        super().__init__(render_mode, config, label)

    def reset(self):
        if self.seed != 'None':
            np.random.seed(self.seed)

        if not self.already_running:
            if self.label is None:
                label = 'default'
            else:
                label = self.label

                # start up sumo
            if self.render_mode == "human":
                print("Creating a sumo-gui.")
                self.sumo_cmd = [sumolib.checkBinary('sumo-gui')]
            else:
                print("No gui will display.")
            self.sumo_cmd.extend(self.arguments)

            traci.start(self.sumo_cmd, label=label)
            self.connection = traci.getConnection(label)
            self.already_running = True
        else:
            self.connection.load(self.arguments)

        # add platoon control plugin in the sumo
        self.plexe = Plexe()
        self.listen_id = self.connection.addStepListener(self.plexe)
        self.count = 0

        # add hdvs on the road
        self.add_random_vehicles()
        c2x = C2X(seed=7005)
        self.platoon = Platoon(self.plexe, c2x)
        self.platoon.build(n=4,
                           speed=40,
                           vType='PlatoonCar',
                           route='freeway',
                           pos=20,
                           lane=0)
        return None

    def step(self, action):
        # map the rl action to pl
        # self._apply_rl_action(action)
        self.platoon.overtaking_step()
        self.count += self.single_step
        self.connection.simulationStep(self.count)

        # An episode is done if the agent has reached the target or crash
        terminated, crash_ids = self._is_done()
        reward = self._get_reward(crash_ids=crash_ids)
        # observation = self._get_obs()
        info = self._get_info(crash_ids=crash_ids)

        return None, reward, terminated, False, info


if __name__ == "__main__":
    with open('Exp_config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    sumo = PLDriving_highway_v1_COOP(config=config, gui=True)
    # graph_obs, mask = sumo.reset()
    reward = 0
    fuel = np.zeros((1, 4))
    obs, info = sumo.reset()
    while True:

        obs, r, dones, _, info = sumo.step(action=None)
        reward += r
        fuel += info['fuel consumption']

        if dones:
            break
    print(reward)
    print(fuel)
