# import sys
# from c2x import C2X
# from platoon import Platoon
# import traci
# import os
# import sumolib
# from plexe import Plexe

# from utils import add_vehicle
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

# sumo_config = './env/cfg/freeway.sumo.cfg'
# arguments = ["--lanechange.duration", "0.5"]
# eval_args = ["-c"]
# # if not evaluate:
# #     eval_args = ["--delay", "0", "-c"]
# arguments.extend(eval_args)
# sumo_cmd = [sumolib.checkBinary('sumo-gui' if True else 'sumo')]
# arguments.append(sumo_config)
# sumo_cmd.extend(arguments)
# traci.start(sumo_cmd)
# plexe = Plexe()
# traci.addStepListener(plexe)
# traci.simulationStep()
# add_vehicle(plexe, "v1", 150, 0, 10, "passenger2")
# traci.vehicle.setSpeedMode("v1", 00000)
# c2x = C2X(seed=7005)
# platoon = Platoon(plexe, c2x)
# platoon.build(n=4,
#               speed=30.55,
#               vType='PlatoonCar',
#               route='route',
#               pos=20,
#               lane=0)
# while True:
#     traci.simulationStep()
#     platoon.overtaking_step()

import sys
from c2x import C2X
from platoon import Platoon
import traci
import os
import sumolib
from plexe import Plexe
import parameter as par


# 添加车辆（hdv）
def add_vehicle(plexe, vid, position, lane, speed, vtype="vtypeauto"):
    if plexe.version[0] >= 1:
        traci.vehicle.add(vid,
                          "freeway",
                          departPos=str(position),
                          departSpeed=str(speed),
                          departLane=str(lane),
                          typeID=vtype)
    else:
        traci.vehicle.add(vid,
                          "freeway",
                          pos=position,
                          speed=speed,
                          lane=lane,
                          typeID=vtype)


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumo_config = '../cfg/freeway_test.sumocfg'
arguments = ["--lateral-resolution", "0.5"]
eval_args = ["-c"]
# if not evaluate:
#     eval_args = ["--delay", "0", "-c"]
arguments.extend(eval_args)
sumo_cmd = [sumolib.checkBinary('sumo-gui' if True else 'sumo')]
arguments.append(sumo_config)
sumo_cmd.extend(arguments)
traci.start(sumo_cmd)
plexe = Plexe()
traci.addStepListener(plexe)
traci.simulationStep()
traci.vehicle.add('v1',
                          "freeway",
                          departPos=120,
                          departSpeed=5,
                          departLane=0,
                          typeID='Car')
traci.vehicle.setSpeedMode('v1',0)
c2x = C2X(seed=7005)
platoon_0 = Platoon(plexe, c2x)
platoon_0.build(
                n=4,
                speed=30.55,
                vType='PlatoonCar',
                route='freeway',
                pos=20,
                lane=0)

while True:
    traci.vehicle.setSpeed('v1',5)
    traci.simulationStep()
    platoon_0.overtaking_step()
