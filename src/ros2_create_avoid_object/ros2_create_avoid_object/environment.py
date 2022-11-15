import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int32
from visualization_msgs.msg import MarkerArray

from environs import Env
import lgsvl

import numpy as np
import torch
import json
import time
import threading
import ctypes
import pandas as pd
import csv
import datetime

MAX_STEPS = 200
NUM_EPISODES = 1000
NUM_PROCESSES = 1
NUM_ADVANCED_STEP = 10
NUM_COMPLETE_EP = 8

os.chdir("/home/chohome/Master_research/LGSVL/ros2_RL_ws/src/ros2_RL_ppo/ros2_RL_ppo")
print("current pose : ", os.getcwd())

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")
now_JST = datetime.datetime.now(JST)
now_time = now_JST.strftime("%Y%m%d%H%M%S")
os.makedirs("./data_{}/weight".format(now_time))


class Environment(Node):
    def __init__(self):
        super().__init__("rl_environment")
        self.env = Env()
        self.sim = lgsvl.Simulator(self.env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host), self.env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
        print("str : ", self.env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host))
        print("int : ", self.env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
        self.simulation_stop_flag = False

        # ROS setting
        self.current_pose = PoseStamped()
        self.imu = Imu()
        self.closest_waypoint = Int32()

        self.current_pose_sub = self.create_subscription(PoseStamped, "current_pose", self.current_poseCallback, 1)
        self.imu_sub = self.create_subscription(Imu, "imu_raw",  self.imuCallback, 1)
        self.closest_waypoint_sub = self.create_subscription(Int32, "closest_waypoint", self.closestWaypointCallback, 1)
        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 1)

        # その他Flagや設定
        self.initialpose_flag = False # ここの値がTrueなら, initialposeによって自己位置が完了したことを示す。
        self.waypoint = pd.read_csv("/home/chohome/Master_research/LGSVL/route/LGSeocho_simpleroute0.5.csv", header=None, skiprows=1).to_numpy()
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, "JST")
        self.now_JST = datetime.datetime.now(JST)
        self.now_time = self.now_JST.strftime("%Y%m%d%H%M%S")
    
    def current_poseCallback(self, msg):
        self.current_pose = msg
        self.initialpose_flag = True
    
    def imuCallback(self, msg):
        self.imu = msg

    def closestWaypointCallback(self, msg):
        self.closest_waypoint = msg.data


    def environment_build(self):
        if self.sim.current_scene == lgsvl.wise.DefaultAssets.LGSeocho:
            self.sim.reset()
        else:
            self.sim.load(lgsvl.wise.DefaultAssets.LGSeocho)
        spawns = self.sim.get_spawn()

        state = lgsvl.AgentState()
        state.transform = spawns[0]
        ego = self.sim.add_agent(self.env.str("LGSVL__VEHICLE_0", lgsvl.wise.DefaultAssets.seniorcar), lgsvl.AgentType.EGO, state)
        print("state : ", state)

        # controllables object setting
        # with open("./warehouse_map_ver2.json") as f:
        #     jsn = json.load(f)
        #     # print("jsn : ", jsn)
        #     for controllables in jsn["controllables"]:
        #         state = lgsvl.ObjectState()
        #         state.transform.position = lgsvl.Vector(controllables["transform"]["position"]["x"],
        #                                                 controllables["transform"]["position"]["y"],
        #                                                 controllables["transform"]["position"]["z"],)
        #         state.transform.rotation = lgsvl.Vector(controllables["transform"]["rotation"]["x"],
        #                                                 controllables["transform"]["rotation"]["y"],
        #                                                 controllables["transform"]["rotation"]["z"],)
        #         state.velocity = lgsvl.Vector(0, 0, 0)
        #         state.angular_velocity = lgsvl.Vector(0, 0, 0)

        #         cone = self.sim.controllable_add(controllables["name"], state)
            
        #     for i, agents in enumerate(jsn["agents"]):
        #         if i > 0:
        #             state = lgsvl.ObjectState()
        #             state.transform.position = lgsvl.Vector(agents["transform"]["position"]["x"],
        #                                                     agents["transform"]["position"]["y"],
        #                                                     agents["transform"]["position"]["z"],)
        #             state.transform.rotation = lgsvl.Vector(agents["transform"]["rotation"]["x"],
        #                                                     agents["transform"]["rotation"]["y"],
        #                                                     agents["transform"]["rotation"]["z"],)
        #             state.velocity = lgsvl.Vector(0, 0, 0)
        #             state.angular_velocity = lgsvl.Vector(0, 0, 0)
        #             # print("car name ]: ", agents["variant"])
        #             npc = self.sim.add_agent(agents["variant"], lgsvl.AgentType.NPC, state=state)

        

        # An EGO will not connect to a bridge unless commanded to
        # print("Bridge connected:", ego.bridge_connected)

        # The EGO is now looking for a bridge at the specified IP and port
        ego.connect_bridge(self.env.str("LGSVL__AUTOPILOT_0_HOST", lgsvl.wise.SimulatorSettings.bridge_host), self.env.int("LGSVL__AUTOPILOT_0_PORT", lgsvl.wise.SimulatorSettings.bridge_port))

        print("Waiting for connection...")

        while not ego.bridge_connected:
            time.sleep(0.1)

        # 車両の初期値点(initial_pose)をpublish
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.stamp = Clock().now().to_msg()
        initial_pose.header.frame_id = "map"
        initial_pose.pose.pose.position.x = -0.0749168
        initial_pose.pose.pose.position.y = 0.04903287
        initial_pose.pose.pose.orientation.z = -0.0179830
        initial_pose.pose.pose.orientation.w = 0.9998382918
        initial_pose.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892326654787]
        self.initialpose_pub.publish(initial_pose)

        # print("Bridge connected:", ego.bridge_connected)
        # print("Running the simulation")
        step = 0
        while not self.simulation_stop_flag: # simulation_stop_flagがFalseのときは実行し続ける
            # print("Runing !!!! step : ", step)
            self.sim.run(0.05)
            step += 1
        # print("SIMULATOR is reseted !!!")
        sys.exit()

    def init_environment(self):
        self.simulation_stop_flag = False
        self.env_thread = threading.Thread(target = self.environment_build, name = "LGSVL_Environment")
        self.env_thread.start()

        self.initialpose_flag = False
        # current_poseが送られてきて自己位置推定が完了したと思われたら self.initalpose_flagがcurrent_pose_callbackの中でTrueになり, whileから抜ける
        while not self.initialpose_flag:
            # print("current_pose no yet ...!")
            time.sleep(0.1)

    def pandas_init(self):
        self.path_record = pd.DataFrame({"current_pose_x" : [self.current_pose.pose.position.x], "current_pose_y" : [self.current_pose.pose.position.y], "current_pose_z" : [self.current_pose.pose.position.z], 
                                    "closest_waypoint_x" : [self.waypoint[self.closest_waypoint][0]], "closest_waypoint_y" : [self.waypoint[self.closest_waypoint][1]], "closest_waypoint_z" : [self.waypoint[self.closest_waypoint][2]],
                                    "lookahead_distance" : [0], "error" : [0]})
            # print("Init path_record : \n", path_record)
    
    def run(self):
        episode_10_array = np.zeros(10) # 10試行分の"経路から外れない", "IMUによる蛇行がしない"step数を格納し, 平均ステップ数を出力に利用

        complete_episodes = 0 # 連続で上手く走行した試行数
        self.closest_waypoint = 1
        episode_final = False # 最後の試行フラグ

        with open('data_{}/episode_mean10_{}.csv'.format(now_time, now_time), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["eisode", "finished_step", "10_step_meaning"])

        self.pandas_init()

        self.init_environment()
        
        episode = 0
        frame = 0