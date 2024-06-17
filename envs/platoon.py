# coding=utf-8
"""
This Python file defines a class called Platoon for simulating and
managing car formations.
It includes functions such as initializing the formation,
checking whether the target lane is unobstructed,
and coordinating communication between formations.

"""

from plexe import FAKED_CACC, ACC, CACC


class CarPlatoon:

    def __init__(self,
                 plexe,
                 num_vehicles,
                 init_positions,
                 init_lane,
                 pl_index,
                 route,
                 connection,
                 lane_count,
                 auto=False,
                 init_speed=20,
                 desire_speed=40,
                 vtype='vtypeauto',
                 cacc_spacing=5,
                 length=5,
                 collaborative_lc=False,
                 safety=False,
                 acc=True,
                 color=[[255, 165, 0], [0, 255, 0]]):
        """
        Initialize a car platoon.

        Arguments:
        - plexe (PLEXE): The PLEXE object.
        - num_vehicles (int): The number of vehicles in the platoon.
        - init_positions (float): The initial position of the first vehicle.
        - init_lane (int): The initial lane of the first vehicle.
        - pl_index (int): The index of the platoon.
        - route (str): The route of the platoon.
        - connection (TraCI): The TraCI connection object.
        - lane_count (int): The number of lanes in the road.
        - auto (bool): Whether the platoon is autonomous.
        - init_speed (float): The initial speed of the platoon.
        - desire_speed (float): The desired speed of the platoon.
        - vtype (str): The vehicle type of the platoon.
        - cacc_spacing (float): The CACC spacing.
        - length (float): The length of the vehicles.
        - collaborative_lc (bool): Whether the platoon uses collaborative lane changing(Plexe).
        - safety (bool): Whether to perform safety checks during lane changes.
        - acc (bool): Whether the platoon uses ACC.
        - color (list): The color of the platoon vehicles.
        """
        # attributes and settings
        self.plexe = plexe
        # attributes
        self.leader_id = 'pl' + '_' + str(pl_index) + '_' + '0'
        self.follower_ids = []
        self.pl_id = 'pl' + '_' + str(pl_index)
        self.desire_speed = desire_speed
        self.cacc_spacing = cacc_spacing
        self.length = length
        self.num_vehicles = num_vehicles
        self.vtype = vtype
        self.route = route
        self.auto = auto
        self.collaborative_lc = collaborative_lc
        self.topology = {}
        self.pl_lane_change = False
        self.target_lane = None
        self.connection = connection
        self.change_safety = safety
        self.lane_count = lane_count - 1
        self.acc = acc

        # add header
        self.connection.vehicle.add(
            self.leader_id,
            self.route,
            typeID=self.vtype,
            departPos=self.num_vehicles * (self.cacc_spacing + self.length) +
            init_positions,
            departSpeed=str(init_speed),
            departLane=str(init_lane),
        )
        self.plexe.set_cc_desired_speed(self.leader_id, self.desire_speed)
        self.plexe.set_acc_headway_time(self.leader_id, 1.5)
        self.connection.vehicle.setColor(
            self.leader_id, (color[0][0], color[0][1], color[0][2]))
        self.plexe.set_active_controller(self.leader_id, ACC)

        if not self.acc:
            self.connection.vehicle.setSpeedMode(self.leader_id, 0)

        if not self.auto:
            self.connection.vehicle.setLaneChangeMode(self.leader_id, 0)
            self.plexe.set_fixed_lane(self.leader_id, init_lane, safe=False)

        assert not (self.auto and collaborative_lc)
        if collaborative_lc:
            self.plexe.enable_auto_lane_changing(self.leader_id, True)

        # add follower
        for i in range(1, self.num_vehicles):
            follower_id = "pl_0_{0}".format(i)
            self.follower_ids.append(follower_id)
            self.connection.vehicle.add(
                follower_id,
                self.route,
                typeID=self.vtype,
                departPos=(self.num_vehicles - i) *
                (self.cacc_spacing + self.length) + init_positions,
                departSpeed=str(init_speed),
                departLane=str(init_lane),
            )
            # xi: damping ratio,omega_n: bandwidth,c1: leader data weighting parameter
            self.plexe.set_path_cacc_parameters(follower_id,
                                                cacc_spacing,
                                                xi=3,
                                                omega_n=0,
                                                c1=0.5)
            if collaborative_lc:
                self.plexe.set_active_controller(follower_id, CACC)
                self.plexe.add_member(self.leader_id, follower_id, i)
            else:
                self.plexe.set_active_controller(follower_id, FAKED_CACC)

            self.plexe.set_fixed_lane(follower_id, init_lane, safe=False)

            self.plexe.set_cc_desired_speed(follower_id, self.desire_speed)
            self.connection.vehicle.setColor(
                follower_id, (color[1][0], color[1][1], color[1][2]))
            self.topology[follower_id] = {
                "front": "{0}_{1}".format(self.pl_id, i - 1),
                "leader": self.leader_id,
            }

            if not self.acc:
                self.connection.vehicle.setSpeedMode(follower_id, 0)

            self.connection.vehicle.setLaneChangeMode(follower_id, 0)
            self.plexe.enable_auto_feed(follower_id, True, self.leader_id,
                                        self.pl_id + "_{0}".format(i - 1))

        # others
        self._buffer = 0

        # frequently lane_change
        self.current_lc_time = 0
        self.last_lc_time = 0

    def lane_change_safety(self, target_lane, dist_threshold):
        # check whether the pl_leader can perform lane_change ensuring safety
        ego_lane = self.connection.vehicle.getLaneIndex(self.leader_id)
        if ego_lane < target_lane:  # turn left
            target_follower = self.connection.vehicle.getNeighbors(
                self.leader_id, 0b000)
            target_leader = self.connection.vehicle.getNeighbors(
                self.leader_id, 0b010)
        elif ego_lane > target_lane:  #turn right
            target_follower = self.connection.vehicle.getNeighbors(
                self.leader_id, 0b001)
            target_leader = self.connection.vehicle.getNeighbors(
                self.leader_id, 0b011)

        follower_safe = (len(target_follower) == 0
                         or target_follower[0][1] >= dist_threshold)
        leader_safe = (len(target_leader) == 0
                       or target_leader[0][1] >= dist_threshold)
        #  if target_follower == () or target_follower[0][1] >= 3:
        #     if target_leader == ():
        #         return True
        #     elif target_leader[0][0] in pl_member:
        #         return True

        return follower_safe and leader_safe

    def __is_target_lane_clear(self, vid, target_lane):
        # Implement the functionality to check if the target lane is clear.
        ego_lane = self.connection.vehicle.getLaneIndex(vid)

        pl_member = [self.leader_id] + self.follower_ids

        if ego_lane < target_lane:  # turn left
            target_follower = self.connection.vehicle.getNeighbors(vid, 0b000)
            target_leader = self.connection.vehicle.getNeighbors(vid, 0b010)
        elif ego_lane > target_lane:  #turn right
            target_follower = self.connection.vehicle.getNeighbors(vid, 0b001)
            target_leader = self.connection.vehicle.getNeighbors(vid, 0b011)

        if len(target_follower) == 0 or target_follower[0][1] >= 3:
            if len(target_leader) == 0 or target_leader[0][
                    0] in pl_member or target_leader[0][1] >= 40:
                return True
        return False

    # convert rl action to  sumo cav action
    def __convert_rl_action(self, rl_action):
        sorted_vehicles = sorted(self.topology.keys())
        pl_member = [self.leader_id] + sorted_vehicles
        ego_lane = self.connection.vehicle.getLaneIndex(self.leader_id)
        if rl_action == 1:
            target_lane = min(self.lane_count, ego_lane + 1)
            if target_lane == ego_lane:
                return
        elif rl_action == 2:
            target_lane = max(0, ego_lane - 1)
            if target_lane == ego_lane:
                return
        elif rl_action == 3:
            for v_id in pl_member:
                ego_speed = self.connection.vehicle.getSpeed(self.leader_id)
                self.connection.vehicle.setSpeed(v_id, ego_speed + 0.1)
            return
        elif rl_action == 4:
            for v_id in pl_member:
                ego_speed = self.connection.vehicle.getSpeed(self.leader_id)
                self.connection.vehicle.setSpeed(v_id, ego_speed - 0.1)
            return
        else:
            return
        return target_lane

    # For communication in platoon, including lane_change action.
    def communicate(self, rl_action):

        if rl_action is None:
            return

        sorted_vehicles = sorted(self.topology.keys())

        # apply header action
        if not self.pl_lane_change:
            # for checking frequently lane change
            self.current_lc_time += 1

            target_lane = self.__convert_rl_action(rl_action)

            if target_lane is None:
                return
            else:
                if self.change_safety:
                    safety = self.lane_change_safety(target_lane,
                                                     dist_threshold=5)
                else:
                    safety = True
                if not safety:
                    return
                else:
                    self.target_lane = target_lane
                    # perform pl_leader lane change action as soon as possible
                    self.connection.vehicle.changeLane(self.leader_id,
                                                       self.target_lane,
                                                       duration=0)
                    self.pl_lane_change = True

                    # perform pl_follower lane change action if allowed
                    for follower_id in sorted_vehicles:
                        if self.change_safety:
                            clear = self.__is_target_lane_clear(
                                follower_id, self.target_lane)
                        else:
                            clear = True
                        if clear:
                            self.connection.vehicle.changeLane(
                                follower_id, self.target_lane, duration=0)
                        else:
                            return
                    self.pl_lane_change = False
                    # frequently lane change
                    self.last_lc_time = self.current_lc_time
                    return

        for follower_id in sorted_vehicles:
            last_lane = self.connection.vehicle.getLaneIndex(follower_id)
            if last_lane != self.target_lane:
                if self.change_safety:
                    clear = self.__is_target_lane_clear(
                        follower_id, self.target_lane)
                else:
                    clear = True
                if clear:
                    self.connection.vehicle.changeLane(follower_id,
                                                       self.target_lane,
                                                       duration=0)
                    if follower_id == self.follower_ids[-1]:
                        self.pl_lane_change = False
                else:
                    return

    # return the current length of the platoon
    def get_length(self):

        front_bumper = self.connection.vehicle.getPosition(self.leader_id)[0]
        if self.follower_ids == []:
            back_bumper = front_bumper
        else:
            back_bumper = self.connection.vehicle.getPosition(
                self.follower_ids[-1])[0] - self.length
        return front_bumper - back_bumper

    # TEST: Experimental functions
    def auto_lane_change(self):
        if self.follower_ids == [] or self.collaborative_lc:
            return
        assert self.auto == True
        leader_lane = self.connection.vehicle.getLaneIndex(self.leader_id)
        follower_lane = self.connection.vehicle.getLaneIndex(
            self.follower_ids[0])
        last_lane = self.connection.vehicle.getLaneIndex(self.follower_ids[-1])
        if leader_lane != last_lane or leader_lane != follower_lane:
            self.connection.vehicle.setLaneChangeMode(self.leader_id, 0)
            # self.plexe.set_fixed_lane(self.leader_id, leader_lane, safe=False)
            self.target_lane = leader_lane
        else:
            self.connection.vehicle.setLaneChangeMode(self.leader_id, 1621)
            # self.plexe.disable_fixed_lane(self.leader_id)
            return

        for follower_id in sorted(self.follower_ids):
            last_lane = self.connection.vehicle.getLaneIndex(follower_id)
            if last_lane != self.target_lane:
                clear = self.__is_target_lane_clear(follower_id,
                                                    self.target_lane)
                if clear:
                    self.connection.vehicle.changeLane(follower_id,
                                                       self.target_lane,
                                                       duration=0)
                else:
                    return

    def simultaneous_lc(self,rl_action):
        # perform lc simultaneously
        target_lane = self.__convert_rl_action(rl_action)

        if target_lane is None:
            return
        self.connection.vehicle.changeLane(self.leader_id,
                                            target_lane,
                                            duration=0)
        sorted_vehicles = sorted(self.topology.keys())
        for follower_id in sorted_vehicles:
            self.connection.vehicle.changeLane(
                    follower_id, target_lane, duration=0)    
