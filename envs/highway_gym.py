"""
This py file defines the env for pl driving on highway in mixed traffic flow.
"""
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import sumolib
import traci

from gymnasium import spaces
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from plexe import Plexe
from sklearn.metrics.pairwise import euclidean_distances

from envs.platoon import CarPlatoon
from coop.src.c2x import C2X
from coop.src.platoon import Platoon


# # use the kinematic information for MLP to make lane-changing strategies
class PLDriving_highway_Kinematic(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode, config, label=None) -> None:
        super().__init__()

        self.config = config

        # sumo
        self.label = label
        self.sumo_config = "envs/cfg/freeway.sumo.cfg"
        self.arguments = [
            "--lanechange.duration", "0.85", "--quit-on-end", "-W"
        ]
        add_args = ["--delay", "0", "-c"]
        self.arguments.extend(add_args)
        self.sumo_cmd = [sumolib.checkBinary('sumo')]
        self.arguments.append(self.sumo_config)
        self.already_running = False

        # env
        self.platoon = []
        self._target_location = 2100
        self.max_speed = self.config['max_speed']
        self.min_speed = 20
        self.single_step = 1  # 1step for 1s simulation
        self.highway_lanes = self.config['highway_lanes']
        self.seed = self.config['seed']

        # reward
        self.w_speed = self.config['w_speed']
        self.w_p_time = self.config['w_p_time']
        self.w_p_crash = self.config['w_p_crash']
        self.w_p_lc = self.config['w_p_lc']

        #crusie, turn left,turn right
        self.n_actions = self.config['n_actions']
        self.action_space = spaces.Discrete(self.n_actions)

        #obs is array containing the surrounding veh state
        self.surrounding_num = 5  # front, 2left, 2 right,head
        F = 3  # speed, pos
        self.observation_space = Box(low=-np.inf,
                                     high=np.inf,
                                     shape=(self.surrounding_num * F, ),
                                     dtype=np.float64)
        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]
        self.render_mode = render_mode

        self.lc_mode = config['lc_mode']
        self.hdv_interval = self.config['hdv_interval']
        self.safe = self.config['safe_monitor']


    # mask the invalid action for MaskablePPO
    def valid_action_mask(self):
        valid_mask = np.ones(
            3, dtype=int)  # corresponding to curise,turn left,turn right

        # turn left or right is invalid when the platoon is changing lane
        if self.pl.pl_lane_change:
            valid_mask[[1, 2]] = 0  
            return valid_mask

        pl_lane_index = self.connection.vehicle.getLaneIndex(self.pl.leader_id)

        # turn left is invalid when the platoon is in the leftmost lane
        # turn right is invalid when the platoon is in the rightmost lane
        if pl_lane_index == 0:
            valid_mask[2] = 0
        elif pl_lane_index == (self.highway_lanes - 1):
            valid_mask[1] = 0

        return valid_mask

    # get the surrounding veh state
    def _get_obs(self):
        surrounding_vehs = []
        current_state = []
        speed_ego = self.connection.vehicle.getSpeed(self.pl.leader_id)
        x_ego, y_ego = self.connection.vehicle.getPosition(self.pl.leader_id)
        speed_diff = []
        x_diff = []
        y_diff = []

        # front right, front left, front, rear right, rear left
        modes = [
            0b000,
            0b001,
            0b011,
            0b010,
        ] 
        for mode in modes:
            veh = self.connection.vehicle.getNeighbors(self.pl.leader_id,
                                                       mode=mode)
            if veh != ():
                surrounding_vehs.append(veh[0][0])
            else:
                surrounding_vehs.append('')
        header = self.connection.vehicle.getLeader(self.pl.leader_id)
        if not header is None:  
            surrounding_vehs.append(header[0])
        else:
            surrounding_vehs.append('')

        # get the relative speed, x, y of the surrounding vehs
        for veh in surrounding_vehs:
            if veh == '':
                x_diff = 0
                y_diff= 0
                speed_diff = 0
            else:
                speed = self.connection.vehicle.getSpeed(veh)
                x, y = self.connection.vehicle.getPosition(veh)
                speed_diff = abs(speed - speed_ego)
                x_diff = abs(x - x_ego)
                y_diff = abs(y - y_ego)
            current_state.append(x_diff)
            current_state.append(y_diff)
            current_state.append(speed_diff)

        return np.array(current_state)


    def _get_info(self, **kwargs):

        crash = True if len(kwargs['crash_ids']) > 0 else False
        return {
            "simulation step": self.count,
            "crash": crash,
        }

    def reset(self, seed=None, options=None):
        if self.seed != 'None':
            np.random.seed(self.seed)

        if not self.already_running:

            # for the case of parallel env
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
        self.hdv_index = 0

        # add car flows on the road
        while self.count < 100:
            self.add_random_flow()
            self.count += self.single_step
            self.connection.simulationStep(self.count)

        self.count += self.single_step + 1
        self.connection.simulationStep(self.count)

        # add platoon on the road
        self.pl = CarPlatoon(self.plexe,
                             num_vehicles=4,
                             init_positions=0,
                             init_lane=0,
                             pl_index=0,
                             route='route',
                             connection=self.connection,
                             safety=self.safe,
                             lane_count=self.highway_lanes)

        self.count += self.single_step
        self.connection.simulationStep(self.count)

        if self.render_mode == "human":
            self.connection.gui.trackVehicle("View #0", self.pl.leader_id)
            self.connection.gui.setZoom("View #0", 1000)

        observation = self._get_obs()
        info = {}

        return observation, info

    def _apply_rl_action(self, action):
        if self.lc_mode == 'simultaneous':
            self.pl.simultaneous_lc(action)
        else:
            self.pl.communicate(action)

    def _is_done(self):
        # crash or leave the road successfully

        done = False

        crash_id = self.connection.simulation.getCollidingVehiclesIDList()
        crash_ids = []
        pl_id = [self.pl.leader_id] + self.pl.follower_ids
        pos = self.connection.vehicle.getPosition(self.pl.leader_id)[0]
        if pos >= self._target_location:
            done = True
            print("{0} success!".format(self.pl.leader_id))
            self.connection.removeStepListener(listenerID=self.listen_id)
        for follower_id in pl_id:
            if follower_id in crash_id:
                done = True
                crash_ids.append(follower_id)
                print('crashing!!!  veh_id:{}'.format(follower_id))
                self.connection.removeStepListener(listenerID=self.listen_id)
        return done, crash_ids

    def _get_reward(self, **kwargs):

        # # including speed reward,crash penalty and time penalty
        unit = 1

        speed_reward = self.connection.vehicle.getSpeed(self.pl.leader_id)

        time = self.connection.vehicle.getDeparture(self.pl.leader_id)
        time_penalty = np.array(self.connection.simulation.getTime() - time)

        # frequently lc penalty
        lc_penalty = 1 if abs(self.pl.current_lc_time - self.pl.last_lc_time) <= 2 else 0

        total_crash_penalty = len(kwargs['crash_ids']) * unit

        reward = self.w_speed * speed_reward - self.w_p_time * time_penalty - self.w_p_lc * lc_penalty - self.w_p_crash * total_crash_penalty
        return np.array(reward)

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
                                        typeID='CarB')
            if self.seed != 'None':
                self.connection.vehicle.setLaneChangeMode(vid, 0)
                self.connection.vehicle.setSpeedMode(vid, 0)
            self.hdv_index += 1

    def step(self, action):
        # map the rl action to pl
        self._apply_rl_action(action)
        self.count += self.single_step
        self.connection.simulationStep(self.count)

        # An episode is done if the agent has reached the target or crash
        terminated, crash_ids = self._is_done()
        reward = self._get_reward(crash_ids=crash_ids)
        observation = self._get_obs()
        info = self._get_info(crash_ids=crash_ids)
        self.add_random_flow()

        return observation, reward, terminated, False, info

    def close(self):
        self.connection.close()

# use the graph data for GNN-based model to make lane-changing strategies
# reference: https://github.com/JiqianDong/GCQ_source
class PLDriving_highway_Graph(PLDriving_highway_Kinematic):

    def __init__(self, render_mode, config, label=None) -> None:
        super().__init__(None, config, label)

        self.config = config

        # graph data setting
        self.N = 90  # maximum veh on the road
        self.N_hdv = 80  #maximum hdv on the road
        self.hdv_length = 5
        self.sense_dist = self.config['sense_dist']


        # obs is a graph structure for GCN etc,which contains
        # node_feat, adjacency, mask

        self.F = 3 + 4  # indicates the number of lanes, [front bumper, back bumper],speed)
        node_feat = Box(low=-np.inf,
                        high=np.inf,
                        shape=(self.N, self.F),
                        dtype=np.float32)
        adjacency = Box(low=0, high=1, shape=(self.N, self.N), dtype=np.int32)
        mask = Box(low=0, high=1, shape=(self.N, ), dtype=np.int32)
        self.action_space = spaces.Discrete(self.n_actions)

        self.observation_space = Dict({
            'node_feat': node_feat,
            'adjacency': adjacency,
            'mask': mask
        })



    def _get_obs(self, hdv_ids):
        """construct a graph for each step
        hdv_ids: hdv ids on the road at current simulation step
        pl_ids: pl ids on the road at current simulation step
        num_lanes:
        sense_dist: sense scope of a cav 
        return: A tuple which including data for construct graph with normalization
        """

        v_ids = hdv_ids + [self.pl.leader_id]
        assert len(hdv_ids) <= self.N_hdv
        node_feat = np.zeros((self.N, self.F), dtype=np.float32)
        adjacency = np.zeros([self.N, self.N], dtype=np.int32)
        mask = np.zeros(self.N, dtype=np.int32)

        # numerical data (speed, location)
        speeds = np.array([(self.connection.vehicle.getSpeed(vid))
                           for vid in v_ids]).reshape(-1, 1)
        front_bumper = np.array([
            self.connection.vehicle.getPosition(vid)[0] for vid in v_ids
        ]).reshape(-1, 1)
        veh_length = np.array([
            self.pl.get_length() if vid.startswith('pl') else self.hdv_length
            for vid in v_ids
        ]).reshape(-1, 1)
        back_bumper = front_bumper - veh_length

        # categorical data
        # 1 hot encoding: lane location encoding
        lanes_column = np.array(
            [self.connection.vehicle.getLaneIndex(vid) for vid in v_ids])
        # lanes = np.zeros([len(v_ids), self.highway_lanes])
        lanes = np.zeros([len(v_ids), 4])  # the max lane index is 4
        lanes[np.arange(len(v_ids)), lanes_column] = 1

        # add the follower laneindex info into pl_leader
        follower_lane_index = self.connection.vehicle.getLaneIndex(
            self.pl.follower_ids[-1])
        lanes[-1, follower_lane_index] = 1

        observed_states = np.c_[front_bumper, back_bumper, speeds, lanes]

        # assemble into the NxF states matrix
        node_feat[:len(hdv_ids), :] = observed_states[:len(hdv_ids), :]
        node_feat[self.N_hdv:self.N_hdv +
                  1, :] = observed_states[len(hdv_ids):, :]

        # normalization
        node_feat[:, 0] /= self._target_location
        node_feat[:, 1] /= self._target_location
        node_feat[:, 2] /= self.max_speed
        # construct the adjacency matrix
        dist_matrix = euclidean_distances(front_bumper)
        adjacency_small = np.zeros_like(dist_matrix)
        adjacency_small[dist_matrix < self.sense_dist] = 1
        adjacency_small[-1:, -1:] = 1

        # assemble into the NxN adjacency matrix

        # adjacency[:len(hdv_ids), :len(hdv_ids)] = adjacency_small[:len(
        #     hdv_ids), :len(hdv_ids)]
        adjacency[self.N_hdv:self.N_hdv + 1, :len(hdv_ids)] = adjacency_small[
            len(hdv_ids):, :len(hdv_ids)]
        adjacency[:len(hdv_ids),
                  self.N_hdv:self.N_hdv + 1] = adjacency_small[:len(hdv_ids),
                                                               len(hdv_ids):]
        adjacency[self.N_hdv:self.N_hdv + 1,
                  self.N_hdv:self.N_hdv + 1] = adjacency_small[len(hdv_ids):,
                                                               len(hdv_ids):]

        # construct the mask
        mask[self.N_hdv:self.N_hdv + 1] = np.ones(1)
        # mask = mask.reshape(-1, 1)
        return {"node_feat": node_feat, "adjacency": adjacency, "mask": mask}

    def reset(self, seed=None, options=None):
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

        self.hdv_index = 0

        while self.count < 100:
            self.add_random_flow()
            self.count += self.single_step
            self.connection.simulationStep(self.count)

        self.count += self.single_step + 1
        self.connection.simulationStep(self.count)

        np.random.seed(None)
        lane_index = np.random.randint(0, self.highway_lanes)
        if self.seed != 'None':
            np.random.seed(self.seed)

        # add platoon on the road
        self.pl = CarPlatoon(
            self.plexe,
            num_vehicles=4,
            init_positions=0,
            # init_lane=0,
            init_lane=lane_index,
            pl_index=0,
            route='route',
            connection=self.connection,
             safety=self.config['safe_monitor'],
            lane_count=self.highway_lanes)

        self.count += self.single_step
        self.connection.simulationStep(self.count)

        if self.render_mode == "human":
            self.connection.gui.trackVehicle("View #0", self.pl.leader_id)
            self.connection.gui.setZoom("View #0", 1000)

        hdv_ids = sorted([
            vid for vid in self.connection.vehicle.getIDList()
            if not vid.startswith('pl')
        ])

        observation = self._get_obs(hdv_ids)
        info = {}

        return observation, info

    def step(self, action):
        # map the rl action to pl
        self._apply_rl_action(action)
        self.count += self.single_step
        self.connection.simulationStep(self.count)
        hdv_ids = sorted([
            vid for vid in self.connection.vehicle.getIDList()
            if not vid.startswith('pl')
        ])
        # An episode is done if the agent has reached the target or crash
        terminated, crash_ids = self._is_done()
        reward = self._get_reward(crash_ids=crash_ids)
        observation = self._get_obs(hdv_ids)
        info = self._get_info(crash_ids=crash_ids)
        self.add_random_flow()

        return observation, reward, terminated, False, info


# use the grid data for CNN-based model to make lane-changing strategies
# reference: https://github.com/Farama-Foundation/HighwayEnv
class PLDriving_highway_OccupancyGrid(PLDriving_highway_Kinematic):

    def __init__(self, render_mode, config, label=None) -> None:
        super().__init__(render_mode, config, label)
        self.features = [
            'presence',
            'x',
            'y',
            'v',
        ]
        self.grid_size = np.array(
            config['grid_size']
        )  # real world size of the grid [[min_x, max_x], [min_y, max_y]]
        self.grid_step = np.array(config['grid_step'])
        self.grid_shape = np.asarray(np.floor(
            (self.grid_size[:, 1] - self.grid_size[:, 0]) / self.grid_step),
                                     dtype=np.int32)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(len(self.features),
                                                   self.grid_shape[0],
                                                   self.grid_shape[1]))


    # get grid data for the whole road
    def _get_obs(self):

        grid = np.zeros((len(self.features), *self.grid_shape))

        v_ids = self.connection.vehicle.getIDList()
        for v_id in v_ids:
            pos = self.connection.vehicle.getPosition(v_id)
            speed = self.connection.vehicle.getSpeed(v_id)
            cell = self.pos_to_index((pos[0]-1, pos[1]))
            grid[0, cell[0], cell[1]] = 1  # persence
            grid[1, cell[0], cell[1]] = pos[0] / 2200
            grid[2, cell[0], cell[1]] = pos[1] / 3 * self.highway_lanes
            grid[3, cell[0], cell[1]] = speed / self.max_speed

        return grid

    def pos_to_index(self, position) -> Tuple[int, int]:
        """
        Convert a world position to a grid cell index

        If align_to_vehicle_axes the cells are in the vehicle's frame, otherwise in the world frame.

        :param position: a world position
        :param relative: whether the position is already relative to the observer's position
        :return: the pair (i,j) of the cell index
        """
        return int(np.floor((position[0] - self.grid_size[0, 0]) / self.grid_step[0])),\
               int(np.floor((position[1] - self.grid_size[1, 0]) / self.grid_step[1]))



class PLDriving_highway_Plexe(PLDriving_highway_Kinematic):

    def __init__(self, render_mode, config, label=None) -> None:
        super().__init__(render_mode, config, label)
        self.collaborative_lc = config['collaborative_lc']
        self.auto = config['auto']

    def reset(self, seed=None, options=None):
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
        self.hdv_index = 0

        while self.count < 100:
            self.add_random_flow()
            self.count += self.single_step
            self.connection.simulationStep(self.count)

        self.count += self.single_step + 1
        self.connection.simulationStep(self.count)

        # add platoon on the road
        self.pl = CarPlatoon(self.plexe,
                             num_vehicles=4,
                             init_positions=0,
                            init_lane=0,
                             pl_index=0,
                             route='route',
                             connection=self.connection,
                             auto=self.auto,
                             collaborative_lc=self.collaborative_lc,
                             lane_count=self.highway_lanes)

        # self.pl.auto_lane_change()
        self.count += self.single_step
        self.connection.simulationStep(self.count)

        if self.render_mode == "human":
            self.connection.gui.trackVehicle("View #0", self.pl.leader_id)
            self.connection.gui.setZoom("View #0", 1000)

        return None, None


    def step(self, action):
        # map the rl action to pl

        self.pl.auto_lane_change()
        self.count += self.single_step
        self.connection.simulationStep(self.count)

        # An episode is done if the agent has reached the target or crash
        terminated, crash_ids = self._is_done()

        reward = self._get_reward(crash_ids=crash_ids)

        info = self._get_info(crash_ids=crash_ids)
        self.add_random_flow()
        return None, reward, terminated, False, info


# use Grid data instead of Kinematic,which can use CNN etc for feature extraction.

# baseline: CoOP: V2V-based Cooperative Overtaking for Platoons on Freeways
# reference: https://github.com/tkn-tub/coop
class PLDriving_highway_CoOP(PLDriving_highway_Kinematic):

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
        self.hdv_index = 0

        while self.count < 100:
            self.add_random_flow()
            self.count += self.single_step
            self.connection.simulationStep(self.count)

        self.count += self.single_step + 1
        self.connection.simulationStep(self.count)

        c2x = C2X(seed=7005)
        self.platoon = Platoon(self.plexe, c2x)

        self.platoon.build(n=4,
                           speed=self.max_speed,
                           vType='PlatoonCar',
                           route='route',
                           pos=0,
                           lane=0)

        if self.render_mode == "human":
            self.connection.gui.trackVehicle("View #0",
                                             self.platoon.leader.v_id)
            self.connection.gui.setZoom("View #0", 1000)

        return None, None

    def _is_done(self):
        # crash or leave the road successfully

        done = False

        crash_id = self.connection.simulation.getCollidingVehiclesIDList()
        crash_ids = []
        pl_id = [self.platoon.leader.v_id] + self.platoon.leader.follower
        pos = self.connection.vehicle.getPosition(self.platoon.leader.v_id)[0]
        if pos >= self._target_location:
            done = True
            print("{0} success!".format(self.platoon.leader.v_id))
            self.connection.removeStepListener(listenerID=self.listen_id)
        for follower_id in pl_id:
            if follower_id in crash_id:
                done = True
                crash_ids.append(follower_id)
                print('crashing!!!  veh_id:{}'.format(follower_id))
                self.connection.removeStepListener(listenerID=self.listen_id)
        return done, crash_ids

    def _get_reward(self, **kwargs):

        # # including speed reward,crash penalty and time penalty
        unit = 1

        speed_reward = self.connection.vehicle.getSpeed(
            self.platoon.leader.v_id)

        time = self.connection.vehicle.getDeparture(self.platoon.leader.v_id)
        time_penalty = np.array(self.connection.simulation.getTime() - time)

        # frequently lc penalty
        lc_penalty = 0

        total_crash_penalty = len(kwargs['crash_ids']) * unit

        reward = self.w_speed * speed_reward - self.w_p_time * time_penalty - self.w_p_lc * lc_penalty - self.w_p_crash * total_crash_penalty
        return reward

    def step(self, action):
        # map the rl action to pl
        # self._apply_rl_action(action)
        self.platoon.overtaking_step()
        self.count += self.single_step
        self.connection.simulationStep(self.count)

        # An episode is done if the agent has reached the target or crash
        terminated, crash_ids = self._is_done()
        reward = self._get_reward(crash_ids=crash_ids)
        info = self._get_info(crash_ids=crash_ids)
        self.add_random_flow()

        return None, reward, terminated, False, info