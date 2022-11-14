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

    def envirionment_build(self):
        if self.sim.current_scene == lgsvl.wise.DefaultAssets.LGSeocho:
            self.sim.reset()
        else:
            self.sim.load(lgsvl.wise.DefaultAssets.LGSeocho)
        spawns = self.sim.get_spawn()

        state = lgsvl.AgentState()
        state.transform = spawns[0]
        ego = self.sim.add_agent(self.env.str("LGSVL__VEHICLE_0", lgsvl.wise.DefaultAssets.seniorcar), lgsvl.AgentType.EGO, state)

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