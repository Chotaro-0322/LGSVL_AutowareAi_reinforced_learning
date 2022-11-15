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

os.chdir("/home/itolab-chotaro/HDD/Master_research/LGSVL/ros2_RL/src/ros2_create_avoid_object/ros2_create_avoid_object")
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
        self.on_collision_flag = False
        self.waypoint = pd.read_csv("/home/itolab-chotaro/HDD/Master_research/LGSVL/route/LGSeocho_toavoid0.5.csv", header=None, skiprows=1).to_numpy()
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
        # 初期化の設定
        self.on_collision_flag = False

        if self.sim.current_scene == lgsvl.wise.DefaultAssets.LGSeocho:
            self.sim.reset()
        else:
            self.sim.load(lgsvl.wise.DefaultAssets.LGSeocho)
        spawns = self.sim.get_spawn()

        state = lgsvl.AgentState()
        state.transform = spawns[0]
        ego = self.sim.add_agent(self.env.str("LGSVL__VEHICLE_0", lgsvl.wise.DefaultAssets.seniorcar), lgsvl.AgentType.EGO, state)

        obstacle_foward = lgsvl.utils.transform_to_forward(spawns[0])
        obstacle_state = lgsvl.AgentState()
        obstacle_state.transform.position = spawns[0].position + 8.0 * obstacle_foward

        obstacle_ego = self.sim.add_agent("Bob", lgsvl.AgentType.PEDESTRIAN, obstacle_state)

        self.objects = {
            ego: "EGO",
            obstacle_ego: "Bob",
        }

        ego.on_collision(self.on_collision)
        obstacle_ego.on_collision(self.on_collision)

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
        # for i in range(5): # 初期値が1度だと飛ばないときがあるので5回飛ばす
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
    
    def on_collision(self, agent1, agent2, contact):
        name1 = self.objects[agent1]
        name2 = self.objects[agent2] if agent2 is not None else "OBSTACLE"
        print("{} collided with {} at {}".format(name1, name2, contact))
        self.on_collision_flag = True
    
    def init_environment(self):
        self.simulation_stop_flag = False
        self.env_thread = threading.Thread(target = self.environment_build, name = "LGSVL_Environment")
        self.env_thread.start()

        self.initialpose_flag = False
        # current_poseが送られてきて自己位置推定が完了したと思われたら self.initalpose_flagがcurrent_pose_callbackの中でTrueになり, whileから抜ける
        while not self.initialpose_flag:
            # print("current_pose no yet ...!")
            time.sleep(0.1)

    def finish_environment(self):
        self.simulation_stop_flag = True # シミュレータを終了させる
        self.env_thread.join()

    def pandas_init(self):
        self.path_record = pd.DataFrame({"current_pose_x" : [self.current_pose.pose.position.x], "current_pose_y" : [self.current_pose.pose.position.y], "current_pose_z" : [self.current_pose.pose.position.z], 
                                    "closest_waypoint_x" : [self.waypoint[self.closest_waypoint][0]], "closest_waypoint_y" : [self.waypoint[self.closest_waypoint][1]], "closest_waypoint_z" : [self.waypoint[self.closest_waypoint][2]],
                                    "lookahead_distance" : [0], "error" : [0]})
            # print("Init path_record : \n", path_record)

    
    def run(self):
        episode_10_array = np.zeros(10) # 10試行分の"経路から外れない", "IMUによる蛇行がしない"step数を格納し, 平均ステップ数を出力に利用
        complete_episodes = np.zeros(10)

        # complete_episodes = 0 # 連続で上手く走行した試行数
        self.closest_waypoint = 1
        episode_final = False # 最後の試行フラグ

        with open('data_{}/episode_mean10_{}.csv'.format(now_time, now_time), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["eisode", "finished_step", "10_step_meaning"])

        self.pandas_init()

        self.init_environment()
        
        episode = 0
        frame = 0

        while True:
            print("self.on_collision_flag : ", self.on_collision_flag)
            if self.closest_waypoint > (len(self.waypoint) - 1 - 10):
                print("reset !!!")
                self.finish_environment()
                time.sleep(3)
                self.init_environment()
            
            if self.on_collision_flag == True:
                print("reset !!!")
                self.finish_environment()
                time.sleep(3)
                self.init_environment()
                
            time.sleep(0.1)