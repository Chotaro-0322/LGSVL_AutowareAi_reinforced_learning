import os
import sys
# print(sys.path)
# sys.path.remove("/opt/ros/melodic/lib/python2.7/dist-packages")
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseStamped, Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Imu, PointCloud2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32, Int32, Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker, MarkerArray

from environs import Env
import lgsvl

from .brain import Actor, Critic, Discriminator, Brain
from .potential_f import Potential_avoid
from .ros2_numpy.ros2_numpy.point_cloud2 import pointcloud2_to_array
from .ros2_numpy.ros2_numpy.occupancy_grid import occupancygrid_to_numpy

import numpy as np
import quaternion
import torch
import json
import time
import threading
import ctypes
import pandas as pd
import csv
import datetime
import threading
from tqdm import tqdm
import glob
import random

MAX_STEPS = 200
NUM_EPISODES = 101
NUM_PROCESSES = 1
NUM_ADVANCED_STEP = 50
NUM_COMPLETE_EP = 8

os.chdir("/home/chohome/Master_research/LGSVL/ros2_RL_ws/src/ros2_RL_potentialavoid_ddpg/ros2_RL_potentialavoid_ddpg")
# os.chdir("/home/itolab-chotaro/HDD/Master_research/LGSVL/ros2_RL/src/ros2_RL_potentialavoid_ddpg/ros2_RL_potentialavoid_ddpg")
print("current pose : ", os.getcwd())

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")
now_JST = datetime.datetime.now(JST)
now_time = now_JST.strftime("%Y%m%d%H%M%S")
os.makedirs("./data_{}/weight".format(now_time))
os.makedirs("./data_{}/image".format(now_time))
json_file_dir = "./data_{}/json".format(now_time)
os.makedirs(json_file_dir)

# Python program raising
# exceptions in a python
# thread

def calculate_dot(rot_matrix, coordinate_array):
    result_array = np.zeros((coordinate_array, 3))
    for i, array in enumerate(coordinate_array):
        result_array[i] = np.dot(rot_matrix, array)
    return result_array

  
class Environment(Node):
    def __init__(self):
        super().__init__("rl_environment")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 環境の準備
        self.env = Env() 
        self.sim = lgsvl.Simulator(self.env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host), self.env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
        print("str : ", self.env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host))
        print("int : ", self.env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
        self.simulation_stop_flag = False

        # ROS setting
        self.current_pose = PoseStamped()
        self.imu = Imu()
        self.closest_waypoint = Int32()

        # self.current_pose_sub = self.create_subscription(PoseStamped, "current_pose", self.current_poseCallback, 1)
        self.imu_sub = self.create_subscription(Imu, "imu_raw",  self.imuCallback, 1)
        self.closest_waypoint_sub = self.create_subscription(Int32, "closest_waypoint", self.closestWaypointCallback, 1)

        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 1)
        self.lookahead_pub = self.create_publisher(Float32, "minimum_lookahead_distance", 1)

        # その他Flagや設定
        self.initialpose_flag = False # ここの値がTrueなら, initialposeによって自己位置が完了したことを示す。
        self.on_collision_flag = False
        self.complete_episode_num = 0
        self.penalty_num = 0
        self.waypoints = pd.read_csv("/home/chohome/Master_research/LGSVL/route/LGSeocho_expert_NOavoid0.5_transformed_ver1.csv", header=None, skiprows=1).to_numpy()
        
        self.base_expert_waypoints = self.waypoints
        # self.waypoints = pd.read_csv("/home/itolab-chotaro/HDD/Master_research/LGSVL/route/LGSeocho_expert_NOavoid0.5_transformed_ver1.csv", header=None, skiprows=1).to_numpy()
        self.global_start = self.waypoints[0].copy()
        self.global_goal = self.waypoints[-1].copy()
        self.expert_come_way_s1_waypoints = glob.glob("/home/chohome/Master_research/LGSVL/route/expert_come_way_s1.0/*.csv")
        self.expert_cross_s05_waypoints = glob.glob("/home/chohome/Master_research/LGSVL/route/expert_cross_s0.5/*.csv")
        self.expert_cross_way_s1_waypoints = glob.glob("/home/chohome/Master_research/LGSVL/route/expert_cross_way_s1.0/*.csv")
        self.expert_data_simple_waypoints = glob.glob("/home/chohome/Master_research/LGSVL/route/expert_data_simple/*.csv")
        self.expert_same_way_s05_waypoints = glob.glob("/home/chohome/Master_research/LGSVL/route/expert_same_way_s0.5/*.csv")
        self.expert_same_way_s15_waypoints = glob.glob("/home/chohome/Master_research/LGSVL/route/expert_same_way_s1.5/*.csv")
        self.scenario = [self.expert_come_way_s1_waypoints, self.expert_cross_s05_waypoints, self.expert_cross_way_s1_waypoints, self.expert_data_simple_waypoints, self.expert_same_way_s05_waypoints, self.expert_same_way_s15_waypoints]
        self.scenario_name = ["expert_come_way_s1.0", "expert_cross_s0.5", "expert_cross_way_s1.0", "expert_data_simple", "expert_same_way_s0.5", "expert_same_way_s1.5"]
        self.map_offset = [43, 28.8, 6.6] # マップのズレを試行錯誤で治す！ ROS[x, y, z] ↔ Unity[z, -x, y]
        self.rotation_offset = [0, 0, 10]
        self.quaternion_offset = quaternion.from_rotation_vector(np.array(self.rotation_offset))
        # print("waypoint : \n", self.waypoint)
        # print("GPU is : ", torch.cuda.is_available())

        self.n_in = 1 # 状態
        self.n_out = 3 #行動
        action_num = 3
        self.n_mid = 32

        self.num_states = 1 # ニューラルネットワークの入力数
        self.num_actions = 2 # purepursuit 0.5mと2m

        self.obs_shape = [100, 100, 1]
        self.actor_up = torch.tensor([1.0, 1.0, 0.0, 0.0, 3.0, 3.0, 0.5, 0.5, 3.0, 3.0, 0.5, 0.5, 100]).to(self.device)
        self.actor_down = torch.tensor([0.0000, 0.0000, -0.0, -0.0, 0.0000, 0.0000, -0.5, -0.5, 0.0000, 0.0000, -0.5, -0.5, 80]).to(self.device)
        self.actor_limit_high = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100])
        self.actor_limit_low = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 80])
        self.actor_value = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 80])
        self.actor = Actor(self.n_in, self.n_mid, self.n_out, self.actor_up, self.actor_down).to(self.device)
        self.critic = Critic(self.obs_shape[2], self.obs_shape[0],self.obs_shape[1]).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.reward_buffer = 0

        self.global_brain = Brain(self.actor, self.critic, self.discriminator, json_file_dir)

        # print("self.obs_shape : ", self.obs_shape)
        self.current_obs = torch.zeros(NUM_PROCESSES, self.obs_shape[2], self.obs_shape[0], self.obs_shape[1]) # torch size ([16, 4])
        self.episode_rewards = torch.zeros([NUM_PROCESSES, 1]) # 現在の施行の報酬を保持
        self.final_rewards = torch.zeros([NUM_PROCESSES, 1]) # 最後の施行の報酬を保持
        self.old_action_probs = torch.zeros([NUM_ADVANCED_STEP, NUM_PROCESSES, 1]) # ratioの計算のため
        self.obs_np = np.zeros([NUM_PROCESSES, self.obs_shape[2], self.obs_shape[0], self.obs_shape[1]]) 
        self.reward_np = np.zeros([NUM_PROCESSES, 1])
        self.done_np = np.zeros([NUM_PROCESSES, 1])
        self.each_step = np.zeros(0) # 各環境のstep数を記録
        self.episode = 0 # 環境0の施行数

        # ポテンシャル法の部分
        self.Potential_avoid = Potential_avoid(delt=0.2, speed=0.5, weight_goal=30) # weight_obstはディープラーニングで計算

        self.pcd_as_numpy_array = np.zeros((0))
        self.gridmap_object_value = np.zeros((0))
        self.current_pose = np.zeros((2))
        self.num_closest_waypoint = 0
        self.gridmap = np.zeros((0))
        self.grid_resolution = 0
        self.grid_width = 0
        self.grid_height = 0
        self.grid_position = Pose()
        self.vehicle_grid = np.zeros((2))
        self.goal_grid = np.zeros((2))

        self.length_judge_obstacle = 10 # この数字分, 先のwaypointを使って障害物を判断

        self.pcd_subscriber = self.create_subscription(PointCloud2, "clipped_cloud", self.pointcloudCallback, 1)
        self.costmap_subscriber = self.create_subscription(OccupancyGrid, "lgsvl_occupancy_grid", self.costmapCallback, 1)
        self.ndt_pose_subscriber = self.create_subscription(PoseStamped, "ndt_pose", self.ndtPoseCallback, 1)
        
        self.waypoint_publisher = self.create_publisher(Float32MultiArray, "route_waypoints_multiarray", 1)
        self.base_waypoint_publisher = self.create_publisher(Float32MultiArray, "base_route_waypoints_multiarray", 1)
        self.lgsvl_obj_publisher = self.create_publisher(Float32MultiArray, "lgsvl_obj", 1)
        self.closest_waypoint_publisher = self.create_publisher(Int32, "closest_waypoint", 1)
        self.npc_marker_publisher = self.create_publisher(MarkerArray, "npc_position", 1)
    
    def imuCallback(self, msg):
        self.imu = msg

    def closestWaypointCallback(self, msg):
        self.closest_waypoint = msg.data
    
    def pointcloudCallback(self, msg):
        self.pcd_as_numpy_array = pointcloud2_to_array(msg)

    def costmapCallback(self, msg):
        self.grid_resolution = msg.info.resolution
        self.grid_width = msg.info.width
        self.grid_height = msg.info.height
        self.grid_origin = msg.info.origin
        self.gridmap_object_value = occupancygrid_to_numpy(msg)[:, :, np.newaxis] # あとで座標情報とくっつけるために次元を増やしておく
        # print("occupancygrid_to_numpy(msg) : \n", occupancygrid_to_numpy(msg).shape)
        orientation_array =  np.array([self.grid_origin.orientation.w, self.grid_origin.orientation.x, self.grid_origin.orientation.y, self.grid_origin.orientation.z])
        orientation_quaternion = quaternion.as_quat_array(orientation_array)
        grid_rot_matrix = quaternion.as_rotation_matrix(orientation_quaternion) # 回転行列の作成
        # print("grid_euler_angle : \n", grid_rot_matrix)
        # 回転行列に平行移動を追加する
        grid_rot_matrix[0, 2] = self.grid_origin.position.x
        grid_rot_matrix[1, 2] = self.grid_origin.position.y

        self.grid_coordinate_array = np.zeros((self.grid_height, self.grid_width, 3))
        self.grid_coordinate_array[:, :, 2] = 1
        for i in range(self.grid_width): # /mapの座標を設定 : x軸 
            self.grid_coordinate_array[:, i, 0] = i * self.grid_resolution
        for i in range(self.grid_height): # /mapの座標を設定 : y軸
            self.grid_coordinate_array[i, :, 1] = i * self.grid_resolution

        # print("grid_rot_matrix : ", grid_rot_matrix.shape)
        # print("self.grid_coordinate : ", self.grid_coordinate_array.shape)
        # 先ほど作成した変形行列(回転+平行移動)を適用する
        self.grid_coordinate_array = self.grid_coordinate_array.reshape(-1, 3)
        
        tmp_transformed_vector = np.zeros((self.grid_coordinate_array.shape[0], 3))
        start = time.time()
        for i, array in enumerate(self.grid_coordinate_array):
            tmp_transformed_vector[i] = np.dot(grid_rot_matrix, array)
        self.grid_coordinate_array = tmp_transformed_vector.reshape(self.grid_height, self.grid_width, 3)

        # print("self.grid_coordinate : \n", self.grid_coordinate_array[:, 99, :])
        self.gridmap = np.block([self.grid_coordinate_array[:, :, :2], self.gridmap_object_value]) # [x, y, class]
        # print("matrix transformed : ", self.gridmap)

    def ndtPoseCallback(self, msg):
        self.current_pose = np.array([msg.pose.position.x, msg.pose.position.y])
        self.initialpose_flag = True

    def environment_build(self):
        if self.sim.current_scene == lgsvl.wise.DefaultAssets.LGSeocho:
            self.sim.reset()
        else:
            self.sim.load(lgsvl.wise.DefaultAssets.LGSeocho)

        transform = lgsvl.Transform(lgsvl.Vector(53.4, 0, -57.6), lgsvl.Vector(0, 0, 0))
        offset = lgsvl.Vector(43, 28.8, 6.6)
        self.sim.set_nav_origin(transform, offset)
        nav_origin = self.sim.get_nav_origin()

        spawns = self.sim.get_spawn()

        state = lgsvl.AgentState()
        obstacle_foward = lgsvl.utils.transform_to_forward(spawns[0])
        obstacle_right = lgsvl.utils.transform_to_right(spawns[0])
        obstacle_up = lgsvl.utils.transform_to_up(spawns[0])
        # state.transform = spawns[0]
        # state.transform.position = spawns[0].position + 8.0 * obstacle_right + 2.0 + obstacle_foward + -2.0 * obstacle_up # シニアカーの位置
        state.transform.position = spawns[0].position + 8.0 * obstacle_right + 2.0 + obstacle_foward + -2.0 * obstacle_up
        
        self.ego = self.sim.add_agent(self.env.str("LGSVL__VEHICLE_0", lgsvl.wise.DefaultAssets.seniorcar), lgsvl.AgentType.EGO, state)

        # どの環境を学習するかここでランダムに取り出す
        # self.expert_num = random.randint(0, len(self.scenario))
        self.expert_num = 0
        self.expert_list = self.scenario[self.expert_num]
        print("self.expert_list : ", self.expert_list)
        obstacle_state = lgsvl.AgentState() 
        
        if self.scenario_name[self.expert_num] == "expert_come_way_s1.0":
            obstacle_state.transform.position = state.transform.position + 12.0 * obstacle_foward# 歩行者が近づいてくる場合の初期値
            obstacle_state.transform.rotation.y = 180 # 歩行者が近づいてくる場合の初期値
            x = -20.0
            y = 0.0
            speed = 1.0
        elif self.scenario_name[self.expert_num] == "expert_cross_s0.5":
            obstacle_state.transform.position = state.transform.position + 6.0 * obstacle_foward + 6.0 * obstacle_right # 歩行者が右から横切る場合の初期位置
            obstacle_state.transform.rotation.y = -90 # 歩行者が右から横切る場合の初期位置
            x = 0.0 # 歩行者が右から横切る場合の初期位置の歩行者の目的地 このときの速度は1.0
            y = -10.0 # 歩行者が右から横切る場合の初期位置の歩行者の目的地 このときの速度は1.0
            speed = 0.5
        elif self.scenario_name[self.expert_num] == "expert_cross_way_s1.0":
            obstacle_state.transform.position = state.transform.position + 6.0 * obstacle_foward + 6.0 * obstacle_right # 歩行者が右から横切る場合の初期位置
            obstacle_state.transform.rotation.y = -90 # 歩行者が右から横切る場合の初期位置
            x = 0.0 # 歩行者が右から横切る場合の初期位置の歩行者の目的地 このときの速度は1.0
            y = -10.0 # 歩行者が右から横切る場合の初期位置の歩行者の目的地 このときの速度は1.0
            speed = 0.8
        elif self.scenario_name[self.expert_num] == "expert_data_simple":
            obstacle_state.transform.position = state.transform.position + 10.0 * obstacle_foward # 歩行者が動かない場合の実験
            x = 0.0
            y = 0.0
            speed = 0.0
        elif self.scenario_name[self.expert_num] == "expert_same_way_s0.5":
            obstacle_state.transform.position = state.transform.position + 3.0 * obstacle_foward
            x = 20
            y = 0.0
            speed = 0.5
        elif self.scenario_name[self.expert_num] == "expert_same_way_s1.5":
            obstacle_state.transform.position = state.transform.position + 3.0 * obstacle_foward
            x = 20
            y = 0.0
            speed = 1.5

        
        self.obstacle_ego = self.sim.add_agent("Bob", lgsvl.AgentType.PEDESTRIAN, obstacle_state)

        # Create wypoints
        pedestrian_waypoints = []
        idle = 1
        
        pedestrian_waypoints.append(lgsvl.WalkWaypoint(obstacle_state.transform.position + x * obstacle_foward + y * obstacle_right , speed=speed, idle=1))
        
        def on_waypoint(agent, index):
            print("Waypoint {} reached".format(index))
        self.obstacle_ego.on_waypoint_reached(on_waypoint)

        self.obstacle_ego.follow(pedestrian_waypoints, True)
        

        self.lgsvl_objects = [
            self.obstacle_ego,
        ]
        self.all_objects = {
            self.ego: "my_vehicle",
            self.obstacle_ego: "obstacle_ego",
        }

        self.ego.on_collision(self.on_collision)
        self.obstacle_ego.on_collision(self.on_collision)

        # An EGO will not connect to a bridge unless commanded to
        # print("Bridge connected:", ego.bridge_connected)

        # The EGO is now looking for a bridge at the specified IP and port
        self.ego.connect_bridge(self.env.str("LGSVL__AUTOPILOT_0_HOST", lgsvl.wise.SimulatorSettings.bridge_host), self.env.int("LGSVL__AUTOPILOT_0_PORT", lgsvl.wise.SimulatorSettings.bridge_port))

        print("Waiting for connection...")

        while not self.ego.bridge_connected:
            time.sleep(0.1)

        # 車両の初期値点(initial_pose)をpublish
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.stamp = Clock().now().to_msg()
        initial_pose.header.frame_id = "map"
        # initial_pose.pose.pose.position.x = -0.0749168
        # initial_pose.pose.pose.position.y = 0.04903287
        # initial_pose.pose.pose.orientation.z = -0.0179830
        # initial_pose.pose.pose.orientation.w = 0.9998382918
        # initial_pose.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892326654787]
        #
        initial_pose.pose.pose.position.x = 1.30901241302
        initial_pose.pose.pose.position.y = -10.1023960114
        initial_pose.pose.pose.orientation.z = -0.0402139551059
        initial_pose.pose.pose.orientation.w = 0.999191091741
        initial_pose.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892326654787]
        self.initialpose_pub.publish(initial_pose)

        # print("Bridge connected:", ego.bridge_connected)
        # print("Running the simulation")
        
        running_count = 0
        markerArray = MarkerArray()
        while not self.simulation_stop_flag: # simulation_stop_flagがFalseのときは実行し続ける
            # print("Runing !!!! step : ", step)
            self.sim.run(0.1)
            if running_count % 2 == 0:
                self.lgsvl_object_publisher()
                # ego_quaternion = quaternion.from_rotation_vector(np.array([ego.state.transform.rotation.x, ego.state.transform.rotation.y, ego.state.transform.rotation.z]))
                # print("ego_quaternion : ", ego_quaternion)
                # NPCの位置を出力
                # marker = Marker()
                # marker.header.frame_id = "/lgsvlmap"
                # marker.type = marker.SPHERE
                # marker.action = marker.ADD
                # marker.scale.x = 1.0
                # marker.scale.y = 1.0
                # marker.scale.z = 1.0
                # marker.color.a = 1.0
                # marker.color.r = 1.0
                # marker.color.g = 1.0
                # marker.color.b = 0.0
                # marker.pose.orientation.x = ego_quaternion.x
                # marker.pose.orientation.y = ego_quaternion.y
                # marker.pose.orientation.z = ego_quaternion.z
                # marker.pose.orientation.w = ego_quaternion.w
                # marker.pose.position.x = self.obstacle_ego.state.transform.position.z + self.map_offset[0]
                # marker.pose.position.y = -self.obstacle_ego.state.transform.position.x + self.map_offset[1]
                # marker.pose.position.z = self.obstacle_ego.state.transform.position.y + self.map_offset[2]

                # markerArray.markers.append(marker)
                # id = 0
                # for m in markerArray.markers:
                #     m.id = id
                #     id += 1
                # self.npc_marker_publisher.publish(markerArray)

                # print("npc_state", type(ego))

            running_count += 1
        # print("SIMULATOR is reseted !!!")
        sys.exit()

    def on_collision(self, agent1, agent2, contact):
        name1 = self.all_objects[agent1]
        name2 = self.all_objects[agent2] if agent2 is not None else "OBSTACLE"
        print("{} collided with {} at {}".format(name1, name2, contact))
        self.on_collision_flag = True

    def publish_route_thread(self):
        while True:
            """------------経路をpublish------------"""
            # print("publish route")
            multiarray = Float32MultiArray()
            multiarray.layout.dim.append(MultiArrayDimension())
            multiarray.layout.dim.append(MultiArrayDimension())
            multiarray.layout.dim[0].label = "height"
            multiarray.layout.dim[1].label = "width"
            multiarray.layout.dim[0].size = self.waypoints.shape[0]
            multiarray.layout.dim[1].size = self.waypoints.shape[1]
            multiarray.layout.dim[0].stride = self.waypoints.shape[0] * self.waypoints.shape[1]
            multiarray.layout.dim[1].stride = self.waypoints.shape[1]
            multiarray.data = self.waypoints.reshape(1, -1)[0].tolist()
            self.base_waypoint_publisher.publish(multiarray)
            time.sleep(0.2)

    def env_feedback(self, actions, episode, first_step = False):
        time.sleep(0.1)
        """------------受け取ったactinosからポテンシャルウェイトを調節-----------------"""
        # self.actor_value += actions
        # self.actor_value = torch.max(torch.min(self.actor_value, self.actor_limit_high), self.actor_limit_low)
        # print("actor_value : \n", self.actor_value)
        """------------経路をpublish------------"""
        # error2waypoint = np.sum(np.abs(self.waypoints[:, :2] - self.current_pose), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
        # # print("error2waypoint : ", error2waypoint)
        # closest_waypoint = error2waypoint.argmin()
        # self.waypoints = self.waypoints[closest_waypoint:]
        # # print("publish route")
        # multiarray = Float32MultiArray()
        # multiarray.layout.dim.append(MultiArrayDimension())
        # multiarray.layout.dim.append(MultiArrayDimension())
        # multiarray.layout.dim[0].label = "height"
        # multiarray.layout.dim[1].label = "width"
        # multiarray.layout.dim[0].size = self.waypoints.shape[0]
        # multiarray.layout.dim[1].size = self.waypoints.shape[1]
        # multiarray.layout.dim[0].stride = self.waypoints.shape[0] * self.waypoints.shape[1]
        # multiarray.layout.dim[1].stride = self.waypoints.shape[1]
        # multiarray.data = self.waypoints.reshape(1, -1)[0].tolist()
        # self.base_waypoint_publisher.publish(multiarray)
        "--------------closest_waypointsおよび、gridmap内の自車位置の予測------------"
        # closest_waypointを探索
        error2waypoint = np.sum(np.abs(self.waypoints[:, :2] - self.current_pose), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
        # print("error2waypoint : ", error2waypoint)
        closest_waypoint = error2waypoint.argmin()
        # print("closest_waypoint : ", closest_waypoint)
        closest_waypoint_msg = Int32()
        closest_waypoint_msg.data = int(closest_waypoint)
        self.closest_waypoint_publisher.publish(closest_waypoint_msg)

        grid_from_currentpose = np.stack([np.full((self.grid_height, self.grid_width), self.current_pose[0])
                                      ,np.full((self.grid_height, self.grid_width), self.current_pose[1])], -1)
        # gridmapとgrid_fram_waypointの差を計算
        error_grid_space = np.sum(np.abs(self.gridmap[:, :, :2] - grid_from_currentpose), axis=2)
        # print("error_grid_scape : ", error_grid_space.shape)
        # 計算された差から, 一番値が近いグリッドを計算
        # nearest_grid_space = [np.argmin(error_grid_space, axis=0)[0], np.argmin(error_grid_space, axis=1)[0]]
        nearest_grid_space = np.unravel_index(np.argmin(error_grid_space), error_grid_space.shape) # 最小値の座標を取得
        # print("nearest_grid_space : ", nearest_grid_space)
        self.vehicle_grid = self.gridmap[nearest_grid_space[0], nearest_grid_space[1]][:2]
        # print("self.vehicle_grid : ", self.vehicle_grid)

        "-------------ポテンシャル法による回避-----------"
        # print("waypoint shape : ", self.waypoints.shape)
        vehicle_position = np.array(self.vehicle_grid)
        goal_position = np.array([self.global_goal[0], self.global_goal[1]]) # 最終点をゴールとして設定 [x, y]
        # print("goal_posistion : ", goal_position)
        # print("current_pose : ", self.current_pose)
        yaw = self.waypoints[closest_waypoint, 3]
        velocity = 6
        change_flag = self.waypoints[0, 5]
        goal_flag, output_route = self.Potential_avoid.calculation(vehicle_position, goal_position, actions, self.gridmap, yaw, velocity, change_flag, now_time, episode, first_step)
        print("output_route : ", output_route.shape)
        # if goal_flag == True:
        # if goal_flag:
        self.waypoints = output_route
        # """------------経路をpublish------------"""
        # # 出力された経路と最も近いウェイポイントを計算
        # error2waypoint = np.sum(np.abs(self.waypoints[:, :2] - self.current_pose), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
        # # print("error2waypoint : ", error2waypoint)
        # closest_waypoint = error2waypoint.argmin()
        # self.waypoints = self.waypoints[closest_waypoint:]
        # # print("publish route")
        # multiarray = Float32MultiArray()
        # multiarray.layout.dim.append(MultiArrayDimension())
        # multiarray.layout.dim.append(MultiArrayDimension())
        # multiarray.layout.dim[0].label = "height"
        # multiarray.layout.dim[1].label = "width"
        # multiarray.layout.dim[0].size = self.waypoints.shape[0]
        # multiarray.layout.dim[1].size = self.waypoints.shape[1]
        # multiarray.layout.dim[0].stride = self.waypoints.shape[0] * self.waypoints.shape[1]
        # multiarray.layout.dim[1].stride = self.waypoints.shape[1]
        # multiarray.data = self.waypoints.reshape(1, -1)[0].tolist()
        # self.base_waypoint_publisher.publish(multiarray)

        "-----------GANの識別器によって人っぽいか人っぽくないかを判別-------------"
        # 経路の正規化(経路waypointsをgridmap)にまとめる)
        generated_route_grid_value = np.zeros((self.grid_height, self.grid_width, 1))
        expert_route_grid_value = np.zeros((self.grid_height, self.grid_width, 1))
        route_grid_xy = np.array(np.meshgrid(np.linspace(-10, 10, num=100), np.linspace(0, 20, num=100))).transpose(1, 2, 0)
        # print("route_grid_xy : ", route_grid_xy.shape)


        # 生成されたresult_routeをgridmapにする.
        for route in self.waypoints:
            grid_from_route = np.stack([np.full((self.grid_height, self.grid_width), route[0])
                                        ,np.full((self.grid_height, self.grid_width), route[1])], -1)
            # print("grid_from_route : ", grid_from_route.shape)
            # gridmapとgrid_fram_waypointの差を計算
            error_grid_space = np.sum(np.abs(route_grid_xy - grid_from_route), axis=2)
            # 計算された差から, 一番値が近いグリッドを計算
            nearest_grid_space = np.unravel_index(np.argmin(error_grid_space), error_grid_space.shape) # 最小値の座標を取得
            # print("nearest_grid_space : ", nearest_grid_space)
            generated_route_grid_value[nearest_grid_space[0], nearest_grid_space[1]] = 1
            # print("self.vehicle_grid : ", self.vehicle_grid)
        
        # エキスパートrouteをgridmapにする.
        # エキスパートデータを自動で選択
        expert_name = random.choice(self.expert_list)
        expert_array = pd.read_csv(expert_name, header=None, skiprows=1).to_numpy()
        sample_waypoints_num = random.choice(range(len(expert_array) - 10)) # 10個前のwaypointsまでをランダムで選択して取り出す
        expert_array = expert_array[sample_waypoints_num:]
        for route in expert_array:
            grid_from_route = np.stack([np.full((self.grid_height, self.grid_width), route[0])
                                        ,np.full((self.grid_height, self.grid_width), route[1])], -1)
            # gridmapとgrid_fram_waypointの差を計算
            error_grid_space = np.sum(np.abs(route_grid_xy - grid_from_route), axis=2)
            # 計算された差から, 一番値が近いグリッドを計算
            nearest_grid_space = np.unravel_index(np.argmin(error_grid_space), error_grid_space.shape) # 最小値の座標を取得
            # print("nearest_grid_space : ", nearest_grid_space)
            expert_route_grid_value[nearest_grid_space[0], nearest_grid_space[1]] = 1
            # print("self.vehicle_grid : ", self.vehicle_grid)

        # 生成されたresult_routeを識別器にかける
        generated_route_grid_value = generated_route_grid_value[:, :, :, np.newaxis]
        # print("generated_route_grid_value : ", generated_route_grid_value.shape)
        generated_route_grid_value = torch.from_numpy(generated_route_grid_value.transpose(3, 2, 0, 1)).type(torch.FloatTensor).to(self.device)
        discriminator_output = self.discriminator(generated_route_grid_value).detach()

        # "生成されたresult_route", "エキスパート経路"を使って識別器を学習させる
        expert_route_grid_value = expert_route_grid_value[:, :, :, np.newaxis]
        # print("export_route : ", expert_route_grid_value.shape)
        expert_route_grid_value = torch.from_numpy(expert_route_grid_value.transpose(3, 2, 0, 1)).type(torch.FloatTensor).to(self.device)
        self.global_brain.discriminator_update(generated_route_grid_value, expert_route_grid_value)

        # 報酬の設定
        dist_vehicle2goal = np.linalg.norm(goal_position - vehicle_position)
        # print("ゴールまでの距離 : ", dist_vehicle2goal)
        discriminator_output = discriminator_output.squeeze().detach().cpu().numpy()
        reward = 0.0
        done = False
        global_final = False

        # 出力された経路と最も近いウェイポイントを計算
        error2waypoint = np.sum(np.abs(self.waypoints[:, :2] - self.current_pose), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
        # print("error2waypoint : ", error2waypoint)
        closest_waypoint = error2waypoint.argmin()
        self.waypoints = self.waypoints[closest_waypoint:]
        # print("publish route")
        multiarray = Float32MultiArray()
        multiarray.layout.dim.append(MultiArrayDimension())
        multiarray.layout.dim.append(MultiArrayDimension())
        multiarray.layout.dim[0].label = "height"
        multiarray.layout.dim[1].label = "width"
        multiarray.layout.dim[0].size = self.waypoints.shape[0]
        multiarray.layout.dim[1].size = self.waypoints.shape[1]
        multiarray.layout.dim[0].stride = self.waypoints.shape[0] * self.waypoints.shape[1]
        multiarray.layout.dim[1].stride = self.waypoints.shape[1]
        multiarray.data = self.waypoints.reshape(1, -1)[0].tolist()
        self.base_waypoint_publisher.publish(multiarray)
        print("行動中...!")

        reward_detail = {"dist_vehicle2goal" : 0, "discriminator_output" : 0, "error2expwaypoint" : 0, "on_collision_flag" : 0, "achive_goal": 0, "goal_flag" : 0}
                
        if dist_vehicle2goal > 1.0: # ゴールに到達できなかった場合
            reward += 0.0
            reward_detail["dist_vehicle2goal"] = 0.0

        if np.round(discriminator_output) == 0: # 識別器によって、生成されたrouteが人っぽいと判断された場合
            if episode > 0: # discriminatorの結果を使用するのは30episode以降
                print("人ではないと判断されました")
                reward -= 1.0
                reward_detail["discriminator_output"] = -1.0
            else:
                reward += 0.0

        error2expwaypoint = np.linalg.norm(self.base_expert_waypoints[:, :2] - self.current_pose, axis=1)
        closest_expwaypoint = error2expwaypoint.argmin()
        if error2expwaypoint[closest_expwaypoint] > 5.0:
            print("エラーが大きくできてしまった")
            reward -= 1.0
            reward_detail["error2expwaypoint"] = -1.0
            done = True
        
        # 障害物に接触した場合
        if self.on_collision_flag:
            print("障害物と接触")
            reward -= 1.0
            reward_detail["on_collision_flag"] = -1.0
            done = True
            self.on_collision_flag = False # 次の準備のためにFalseに変更しておく

        # ゴールに到達した場合
        if np.linalg.norm(goal_position - self.current_pose) < 4.0:
            print("ゴールしました")
            reward += 1.0
            reward_detail["achive_goal"] = 1.0
            self.complete_episode_num += 1
            done = True
        else:
            self.complete_episode_num = 0
        
        if self.complete_episode_num >= 10:
            print("10回ゴール [完全終了]")
            global_final = True

        if goal_flag is not True: # ゴールできないような経路を作成してしまった場合
            print("ゴールまでの経路を作成できなかった")
            reward -= 1.0
            reward_detail["goal_flag"] = -1.0
            # done = True

        # if reward > 0:
        #     reward = 1.0
        # elif reward < 0:
        #     reward = -1.0
        # else:
        #     reward = 0.0

        return torch.FloatTensor([reward]), done, global_final, reward_detail

    def init_environment(self):
        # 速度を0にした仮想のwayointをpublishして勝手に前のwaypointsを使って走り出すのを防ぐ
        self.waypoints = np.array([[self.global_start[0], self.global_start[1], self.global_start[2], self.global_start[3], 0, self.global_start[5]]])
        print("init_waypoints : ", self.waypoints.shape)
        multiarray = Float32MultiArray()
        multiarray.layout.dim.append(MultiArrayDimension())
        multiarray.layout.dim.append(MultiArrayDimension())
        multiarray.layout.dim[0].label = "height"
        multiarray.layout.dim[1].label = "width"
        multiarray.layout.dim[0].size = self.waypoints.shape[0]
        multiarray.layout.dim[1].size = self.waypoints.shape[1]
        multiarray.layout.dim[0].stride = self.waypoints.shape[0] * self.waypoints.shape[1]
        multiarray.layout.dim[1].stride = self.waypoints.shape[1]
        multiarray.data = self.waypoints.reshape(1, -1)[0].tolist()
        self.base_waypoint_publisher.publish(multiarray)
        time.sleep(0.2)
        self.penalty_num = 0
        self.simulation_stop_flag = False
        self.env_thread = threading.Thread(target = self.environment_build, name = "LGSVL_Environment")
        self.env_thread.start()

        self.initialpose_flag = False
        # current_poseが送られてきて自己位置推定が完了したと思われたら self.initalpose_flagがcurrent_pose_callbackの中でTrueになり, whileから抜ける
        
        while not self.initialpose_flag or self.gridmap.shape[0] == 0:
            # print("self.gridmap_length : ", self.gridmap.shape[0])
            # print("current_pose no yet ...!")
            time.sleep(0.1)
        # print("self.gridmap_length : ", self.gridmap.shape[0])

    def finish_environment(self):
        self.simulation_stop_flag = True # シミュレータを終了させる
        self.env_thread.join()

    def pandas_init(self):
        self.record = pd.DataFrame({"current_pose_x" : [self.current_pose[0]], "current_pose_y" : [self.current_pose[1]], "current_pose_z" : [0],
                                        "reward" : [0], "reward_mean" : [0], 
                                        "action_wall_x" : [0], "action_wall_y" : [0],
                                        "action_vehicle_x ": [0], "action_vehicle_y ": [0],
                                        "action_human_x" :  [0], "action_human_y" :  [0],
                                        "dist_vehicle2goal" : [0], "discriminator_output" : [0],
                                        "error2expwaypoint" : [0], "on_collision_flag" : [0],
                                        "achive_goal": [0], "goal_flag" : [0]})
            # print("Init path_record : \n", path_record)

    def lgsvl_object_publisher(self):
        object_array = np.zeros((len(self.lgsvl_objects), 3))
        # print("self.lgesvl_object : \n", self.lgsvl_objects[0].state.transform.position.z)
        for i, obj in enumerate(self.lgsvl_objects):
            # print("obj : ", type(obj))
            if isinstance(obj, lgsvl.agent.NpcVehicle):
                # print("vehicle !!!")
                obj_x = obj.state.transform.position.z + self.map_offset[0]
                obj_y = -obj.state.transform.position.x + self.map_offset[1]
                obj_class = 1
            elif isinstance(obj, lgsvl.agent.Pedestrian):
                obj_x = obj.state.transform.position.z + self.map_offset[0]
                obj_y = -obj.state.transform.position.x + self.map_offset[1]
                obj_class = 2
            else:
                obj_x = 0
                obj_y = 0
                obj_class = 0
            object_array[i, 0:3] = [obj_x, obj_y, obj_class]

        multiarray = Float32MultiArray()
        multiarray.layout.dim.append(MultiArrayDimension())
        multiarray.layout.dim.append(MultiArrayDimension())
        multiarray.layout.dim[0].label = "height"
        multiarray.layout.dim[1].label = "width"
        multiarray.layout.dim[0].size = object_array.shape[0]
        multiarray.layout.dim[1].size = object_array.shape[1]
        multiarray.layout.dim[0].stride = object_array.shape[0] * object_array.shape[1]
        multiarray.layout.dim[1].stride = object_array.shape[1]
        multiarray.data = object_array.reshape(1, -1)[0].tolist()

        # print("lgsvl is published ")
        self.lgsvl_obj_publisher.publish(multiarray)

    def run(self):
        episode_10_array = np.zeros(10) # 10試行分の"経路から外れない", "IMUによる蛇行がしない"step数を格納し, 平均ステップ数を出力に利用
        complete_episodes = np.zeros(10)

        # complete_episodes = 0 # 連続で上手く走行した試行数
        self.closest_waypoint = 1
        episode_final = False # 最後の試行フラグ

        global_record = pd.DataFrame({"episode" : [0], "step_sum_reward" : [0], "result_step" : [0],
                                      "states" : "dafault", "action_min" : [float(self.actor_down[0])], "action_max" : [float(self.actor_up[0])],
                                      "map_min" : [float(self.actor_down[-1])], "map_max" : [float(self.actor_up[-1])]})

        self.pandas_init()

        self.init_environment()

        # print("self.gridmap_object_value : ", self.gridmap_object_value.shape)
        episode = 0
        frame = 0

        for episode in tqdm(range(NUM_EPISODES)):
            state_stack = torch.zeros(4, self.grid_height, self.grid_width)
            for i in range(4):
                state_stack[i] = torch.from_numpy(self.gridmap_object_value.transpose(2, 0, 1))
            state_stack = torch.unsqueeze(state_stack, 0).type(torch.FloatTensor).to(self.device)
            
            reward_list = np.zeros(0)
            for step in range(MAX_STEPS):          
                with torch.no_grad():
                    state_stack = torch.zeros(4, self.grid_height, self.grid_width)
                    for i in range(4):
                        state_stack[i] = torch.from_numpy(self.gridmap_object_value.transpose(2, 0, 1))
                    state_stack = torch.unsqueeze(state_stack, 0).type(torch.FloatTensor).to(self.device)
                    action, _ = self.actor.act(state_stack, episode)
                actions = action[0].cpu().type(torch.FloatTensor)
                # print("actions : \n", actions)
                
                first_step = False
                if step == 0: # 最初のステップならポテンシャル場を描写するためのfirst_step==Trueにする
                    first_step = True
                reward, done, global_final, reward_detail = self.env_feedback(actions=actions, episode=episode, first_step=first_step) # errorを次のobsercation_next(次の状態)として扱う
                observation_stack = torch.zeros(4, self.grid_height, self.grid_width)
                for i in range(4): # 次の状態を格納
                    observation_stack[i] = torch.from_numpy(self.gridmap_object_value.transpose(2, 0, 1))
                observation_stack = torch.unsqueeze(observation_stack, 0).type(torch.FloatTensor).to(self.device)
                # rewardの格納 & 10回分の平均を計算
                reward_list = np.append(reward_list, reward)
                mean_reward = np.mean(reward_list)
                
                # データをcsvに保存
                self.record = self.record.append({"current_pose_x" : self.current_pose[0], "current_pose_y" : self.current_pose[1], "current_pose_z" : 0,
                                                "reward" : float(reward), "reward_mean" : mean_reward, 
                                                "action_wall_x" : float(actions[0]), "action_wall_y" : float(actions[0]),
                                                "action_vehicle_x ": float(actions[2]), "action_vehicle_y ": float(actions[3]),
                                                "action_human_x" :  float(actions[4]), "action_human_y" :  float(actions[5]), 
                                                "dist_vehicle2goal" : reward_detail["dist_vehicle2goal"], "discriminator_output" : reward_detail["discriminator_output"], 
                                                "error2expwaypoint" : reward_detail["error2expwaypoint"], "on_collision_flag" : reward_detail["on_collision_flag"], 
                                                "achive_goal": reward_detail["achive_goal"], "goal_flag" : reward_detail["goal_flag"]}, ignore_index=True)

                if done: # simulationが止まっていなかったらFalse, 終了するならTrue
                    next_state = observation_stack # 便宜上, 0とおいておく. あとで省きます

                    episode_10_array[episode % 10] = frame
                    
                    if episode % 10 == 0: # 10episodeごとにウェイトを保存
                        torch.save(self.global_brain.main_actor, "./data_{}/weight/episode_{}_finish.pth".format(now_time, episode))

                    if reward <= -1: # 途中で上手く走行できなかったら罰則として-1
                        print("[失敗]報酬 : - 1")
                        complete_episodes[episode % 10] = 0 # 連続成功記録をリセット
                        self.record.to_csv("./data_{}/learning_log_{}_ep{}_failure.csv".format(now_time, now_time, episode))
                        result = "failure"
                    
                    elif reward >= 1: # 何事もなく、上手く走行出来たら報酬として+1
                        print("[成功]報酬 : + 1")
                        complete_episodes[episode % 10] = 1 # 連続成功記録を+1
                        self.record.to_csv("./data_{}/learning_log_{}_ep{}_success.csv".format(now_time, now_time, episode))
                        result = "success"

                    # episode += 1
                    frame = 0

                else:
                    self.each_step += 1
                    next_state = observation_stack
                print("next_state : ", next_state.size())
                        
                frame += 1
                # if reward >= 1:
                #     print("+1 報酬！！ : ", reward)
                json_t_delta = datetime.timedelta(hours=9)
                json_JST = datetime.timezone(json_t_delta, "JST")
                json_now_JST = datetime.datetime.now(json_JST)
                json_now_time = json_now_JST.strftime("%Y%m%d%H%M%S")
                self.global_brain.memory.push(state_stack, action, next_state, reward, done, json_now_time)

                self.global_brain.td_memory.push(0)

                self.global_brain.actorcritic_update(step, episode)

                state = next_state
                # print("new state : ", state.size())

                if done:
                    # 分散処理の最初の項が終了した場合、結果を記録
                    self.finish_environment()
                    print("%d Episode: Finished after %d frame : 10試行の平均frame数 = %.1lf" %(episode, frame, episode_10_array.mean()))
                    # with open('data_{}/episode_mean10_{}.csv'.format(now_time, now_time).format(), 'a') as f:
                    #     writer = csv.writer(f)
                    #     writer.writerow([episode, frame, episode_10_array.mean()])
                    # self.global_brain.td_memory.update_td_error()

                    # if (episode % 2 == 0):
                        # self.global_brain.update_target_q_function()
                    
                    global_record = global_record.append({"episode" : episode, "step_sum_reward" : np.sum(reward_list), 
                                                          "result_step" : step, "states" : result}, ignore_index=True)
                    
                    global_record.to_csv("./data_{}/global_log.csv".format(now_time))

                    self.pandas_init()
                    # done がTrueのとき、環境のリセット(分散学習のとき、env[i].reset()みたいなことをしないといけない. その場合、ワークステーション10台くらい必要)
                    self.finish_environment()
                    time.sleep(3)
                    self.init_environment()
                    break
            
            if global_final:
                print("10回連続成功")
                
                torch.save(self.global_brain.actor_critic, "./data_{}/weight/episode_{}_finish.pth".format(now_time, episode))
                print("無事に終了")
                episode_final = True
                break
                
                
            





