import numpy as np
import traci
import sumolib

import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete,Box

from pettingzoo import ParallelEnv

class SumoPettingZooEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None,seed=0):
        super().__init__()
        self.sumo_config = "envs/cfg/freeway.sumo.cfg"
        self.arguments = ["--lanechange.duration", "0.85", "--quit-on-end"]
        add_args = ["--delay", "100", "-c"]
        self.arguments.extend(add_args)
        self.sumo_cmd = sumolib.checkBinary('sumo')
        self.sumo_cmd_gui = sumolib.checkBinary('sumo-gui')
        self.arguments.append(self.sumo_config)
        self.already_running = False

        self._target_location = 2100
        self.max_speed = 40
        self.min_speed = 20
        self.single_step = 1
        self.highway_lanes = 4
        self.seed = seed
        self.hdv_interval = 2

        self.headway_time = 1.2


        self.w_p_crash = 200
        self.w_speed = 1
        self.w_p_headway = 4
        self.w_follow = 5
        
        self.last_lane_change_time = {}  # 记录每辆车上一次换道时间


        self.n_actions = 5
        self.count = 0
        
        self.possible_agents = ["cav_0", "cav_1", "cav_2", "cav_3"]
        
        self.render_mode = render_mode

    def add_random_flow(self):
        if self.count % self.hdv_interval == 0:
            vid = 'hdv_' + str(self.hdv_index)
            speed = np.random.randint(20, 30)
            lane_index = np.random.randint(0, self.highway_lanes)
            self.connection.vehicle.add(vid,
                                        "route",
                                        departPos=0,
                                        departSpeed=speed,
                                        departLane=lane_index,
                                        typeID='vtypeA')
            if self.seed != 'None':
                # self.connection.vehicle.setSpeed(vid,speed)
                self.connection.vehicle.setLaneChangeMode(vid, 0)
                # self.connection.vehicle.setSpeedMode(vid, 0)
            self.hdv_index += 1

    def reset(self):
        self.agents = copy(self.possible_agents)
        self.last_lane_change_time = {}
        
        # start sumo and initialize vehicles
        if self.seed != 'None':
            np.random.seed(self.seed)

        if not self.already_running:
            sumo_cmd = [self.sumo_cmd_gui if self.render_mode == "human" else self.sumo_cmd]
            sumo_cmd.extend(self.arguments)
            traci.start(sumo_cmd)
            self.connection = traci
            self.already_running = True
        else:
            self.connection.load(self.arguments)

        self.count = 0
        self.hdv_index = 0

        while self.count < 100:
            self.add_random_flow()
            self.count += self.single_step
            self.connection.simulationStep(self.count)

        self.count += self.single_step + 1
        self.connection.simulationStep(self.count)
        
        # Initialize control vehicles
        self.finish_id = []
        for i, agent in enumerate(self.agents):
            init_position= 0

            self.connection.vehicle.add(agent, "route", departPos=(len(self.agents) - i) *
                (5 + 5) + init_position, departSpeed=self.min_speed, departLane=0, typeID='vtypeB')
            self.connection.vehicle.setSpeed(agent, self.min_speed)
            self.connection.vehicle.setSpeedMode(agent, 0)
            self.connection.vehicle.setLaneChangeMode(agent, 0)

        self.count += self.single_step
        self.connection.simulationStep(self.count)

        if self.render_mode == "human":
            self.connection.gui.trackVehicle("View #0", self.agents[0])
            self.connection.gui.setZoom("View #0", 1000)


        observations = np.array([self._get_obs(agent) for agent in self.agents])
        infos = {a: {} for a in self.agents}


        return observations,infos

    def _get_obs(self, agent):
        surrounding_vehs = []
        current_state = []
        surrounding_vehs.append(agent)
        modes = [
            0b000,
            0b001,
            0b011,
            0b010,
        ]  #left,right
        for mode in modes:
            veh = self.connection.vehicle.getNeighbors(agent,
                                                       mode=mode)
            if veh != ():
                surrounding_vehs.append(veh[0][0])
            else:
                surrounding_vehs.append('')
        header = self.connection.vehicle.getLeader(agent)
        if not header is None:  # 前车
            surrounding_vehs.append(header[0])
        else:
            surrounding_vehs.append('')
        # backer = self.connection.vehicle.getFollower(self.pl.follower_ids[-1])
        # surrounding_vehs.append(backer[0])
        for veh in surrounding_vehs:
            if veh == '':
                x = 0
                y = 0
                speed = 0
                present = 0
            else:
                speed = self.connection.vehicle.getSpeed(veh)
                x, y = self.connection.vehicle.getPosition(veh)
                present = 1
            current_state.append(x)
            current_state.append(y)
            current_state.append(speed)
            current_state.append(present)
        return np.array(current_state)
    
    def _compute_headway_distance(
        self,
        agent,
    ):
        headway_distance = 2 * self.max_speed
        header = self.connection.vehicle.getLeader(agent)
        if not header is None:  # 前车
            if header[1] < headway_distance and header[1]>0:
                headway_distance = header[1]
            else:
                headway_distance = 0.01

        return headway_distance

    def _agent_reward(self, i) -> float:
        """
            The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
            But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
            :param action: the action performed
            :return: the reward of the state-action transition
       """
        ego_vehicle = self.agents[i]

        # compute high speed reward
        ego_speed = self.connection.vehicle.getSpeed(ego_vehicle)
        ego_position = self.connection.vehicle.getPosition(ego_vehicle)

        R_os = (ego_speed - self.min_speed) / (
            self.max_speed - self.min_speed)

        # compute crash penalty
        if ego_vehicle in self.crash_id:
            R_c = -1
        else:
            R_c = 0

        #compute headway penalty
        headway_distance = self._compute_headway_distance(ego_vehicle)
        R_h = np.log(headway_distance /
                     (self.headway_time *
                      ego_speed)) if ego_speed > 0 else 0

        # compute follow reward
        R_f = 0
        if i != 0:
            front_vehicle = self.agents[i - 1]
            front_position = self.connection.vehicle.getPosition(front_vehicle)
            if front_position[1] == ego_position[1]:
                R_f = 0.7 * 0.3
            elif (front_position[0] -
                  ego_position[0]) <= 2 * self.max_speed:
                R_f = 0.3 * 0.25 * (
                    front_position[0] - ego_position[0]) / (
                        self.headway_time * self.max_speed)

        # compute overall reward
        reward = self.w_p_crash * R_c \
                 + self.w_speed * R_os \
                 + self.w_follow * R_f \
                 + self.w_p_headway * R_h
        return reward

    def _regional_reward(self):
        surrounding_vehs = []
        neighbor_vehicle = []
        regional_rewards = {}
        for agent in self.agents:
            surrounding_vehs.append(agent)
            modes = [
                0b000,
                0b001,
                0b011,
                0b010,
            ]  #left,right
            for mode in modes:
                veh = self.connection.vehicle.getNeighbors(agent,
                                                        mode=mode)
                
                if veh != ():
                    surrounding_vehs.append(veh[0][0])
            header = self.connection.vehicle.getLeader(agent)
            if not header is None:  # 前车
                surrounding_vehs.append(header[0])

            for v in surrounding_vehs:
                if 'cav' in v:
                    neighbor_vehicle.append(v)
            regional_reward = sum(self.local_rewards[v] for v in neighbor_vehicle)
            regional_rewards[agent] = regional_reward / len(neighbor_vehicle)
        return regional_rewards
    
    def _reward(self) -> float:
        # Cooperative multi-agent reward
        local_rewards = {}
        for i,agent in enumerate(self.agents):
            reward = self._agent_reward(i)
            local_rewards[agent] = reward

        return local_rewards

    def _reward_metric(self) -> float:
        # 统一baseline与自身模型的度量
        # platoon_follow collision_avoid speed_reward time_cost
        r_metic ={}

        for agent in self.agents:
            speed_reward = self.connection.vehicle.getSpeed(agent)

            time = self.connection.vehicle.getDeparture(agent)
            current_time = self.connection.simulation.getTime()
            time_penalty = np.array(current_time - time)

            # frequently lc penalty
            lc_penalty = 0

            if agent in self.last_lane_change_time:
                last_change_time = self.last_lane_change_time[agent]
                if current_time - last_change_time < 2:  # 在两秒内有连续换道动作
                    lc_penalty = -1 

            if agent in self.crash_id:
                R_c = -1
            else:
                R_c = 0

            reward = 1 * speed_reward - 0.3 * time_penalty - 0.1  * lc_penalty - 100 * R_c
            r_metic[agent] = reward
        return r_metic

    def _update_lane_change_time(self, agent):
        current_time = traci.simulation.getTime()
        self.last_lane_change_time[agent] = current_time

    def step(self, actions: np.array):
        infos = {}
    
        for index, action in enumerate(actions):
            self._apply_action(index, action)

        self.count += self.single_step
        self.connection.simulationStep(self.count)
        
        # carsh or not
        self.crash_id = self.connection.simulation.getCollidingVehiclesIDList()

        # local reward
        self.local_rewards = self._reward()
        infos['reward_metric'] = self._reward_metric()
        # regional reward
        regional_rewards =  self._regional_reward()
        infos["regional_rewards"] = tuple(
            regional_rewards[agent] for agent in self.agents) 
        # global reward
        if self.local_rewards!= {}:
            reward = sum(list(self.local_rewards.values())) / len(self.agents)
        else:
            reward = 0
        infos["agents_dones"] = tuple(
            self._agent_is_terminal(agent)
            for agent in self.agents)
        done = self._is_terminated() or self._is_truncated()

        observations = np.array([self._get_obs(agent) for agent in self.agents])

        self.add_random_flow()

        return observations, reward, done, infos

    def _apply_action(self, index, action):
        # Apply the action to the specified agent
        if action == 0:  # accelerate
            if self.connection.vehicle.getSpeed(self.agents[index]) < self.max_speed:
                self.connection.vehicle.setAcceleration(self.agents[index], 0.5,duration=self.single_step)
        elif action == 1:  # decelerate
            if self.connection.vehicle.getSpeed(self.agents[index]) > self.min_speed:
                self.connection.vehicle.setAcceleration(self.agents[index], -0.5,duration=self.single_step)
        elif action == 2:  # change lane left
            ego_lane = self.connection.vehicle.getLaneIndex(self.agents[index])
            target_lane = min(self.highway_lanes-1, ego_lane + 1)
            if target_lane == ego_lane:
                return
            self.connection.vehicle.changeLane(self.agents[index], target_lane,duration=0)
            self._update_lane_change_time(self.agents[index])
        elif action == 3:  # change lane right
            ego_lane = self.connection.vehicle.getLaneIndex(self.agents[index])
            target_lane = max(0, ego_lane - 1)
            if target_lane == ego_lane:
                return
            self.connection.vehicle.changeLane(self.agents[index], target_lane,duration=0)
            self._update_lane_change_time(self.agents[index])
        elif action == 4:  # keep lane
            pass

    def _agent_is_terminal(self,agent):
        crash = agent in self.crash_id
        finish = False
        if self.connection.vehicle.getPosition(agent)[0] > self._target_location:
            self.finish_id.append(agent)
            self.connection.vehicle.remove(agent)
            self.agents.remove(agent)           
        return crash or finish 

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        crash = any(agent in self.crash_id for agent in self.agents)

        finish = False
        if len(self.finish_id) == 4:
            finish = True

        return crash or finish

    def _is_truncated(self) -> bool:
        truncated = False
        if self.count >= 300:
            truncated = True
        return truncated

    def render(self):
        pass

    def sample(self):
        actions = {}
        for agent_id in self.agents:
            actions[agent_id] = self.action_space().sample()
        return actions

    def close(self):
        traci.close()
        self.already_running = False

    def observation_space(self):
        surrounding_num = 6
        F = 4
        observation_space =Box(low=-np.inf, high=np.inf, shape=(len(self.possible_agents),surrounding_num * F ), dtype=np.float64)
        return observation_space
    
    def action_space(self):
        return Discrete(5)
    

if __name__ == "__main__":
    env = SumoPettingZooEnv(render_mode="human")
    env.reset()
    for _ in range(50):
        action = env.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break