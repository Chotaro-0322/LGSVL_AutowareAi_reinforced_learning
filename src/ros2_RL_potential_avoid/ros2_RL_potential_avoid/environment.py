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

from .brain import Actor, Critic, Discriminator, Brain, RolloutStorage
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

MAX_STEPS = 200
NUM_EPISODES = 1000
NUM_PROCESSES = 1
NUM_ADVANCED_STEP = 10
NUM_COMPLETE_EP = 8

#os.chdir("/home/chohome/Master_research/LGSVL/ros2_RL_ws/src/ros2_RL_ppo/ros2_RL_ppo")
os.chdir("/home/itolab-chotaro/HDD/Master_research/LGSVL/ros2_RL/src/ros2_RL_potential_avoid/ros2_RL_potential_avoid")
print("current pose : ", os.getcwd())

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")
now_JST = datetime.datetime.now(JST)
now_time = now_JST.strftime("%Y%m%d%H%M%S")
os.makedirs("./data_{}/weight".format(now_time))

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
        # self.waypoint = pd.read_csv("/home/chohome/Master_research/LGSVL/route/LGSeocho_simpleroute0.5.csv", header=None, skiprows=1).to_numpy()
        self.waypoints = pd.read_csv("/home/itolab-chotaro/HDD/Master_research/LGSVL/route/LGSeocho_toavoid0.5.csv", header=None, skiprows=1).to_numpy()
        self.global_start = self.waypoints[0].copy()
        self.global_goal = self.waypoints[-1].copy()
        self.expert_waypoints = pd.read_csv("/home/itolab-chotaro/HDD/Master_research/LGSVL/route/LGSeocho_expert_avoid0.5_ver2.csv", header=None, skiprows=1).to_numpy()
        self.expert_global_start = self.expert_waypoints[0].copy()
        self.expert_global_goal = self.expert_waypoints[-1].copy()
        self.map_offset = [43, 28.8, 6.6] # マップのズレを試行錯誤で治す！ ROS[x, y, z] ↔ Unity[z, -x, y]
        self.rotation_offset = [0, 0, 10]
        self.quaternion_offset = quaternion.from_rotation_vector(np.array(self.rotation_offset))
        # print("waypoint : \n", self.waypoint)
        # print("GPU is : ", torch.cuda.is_available())

        self.n_in = 1 # 状態
        self.n_out = 3 #行動
        action_num = 3
        self.n_mid = 32

        self.actor_hight = torch.tensor([0.2]).to(self.device)
        self.actor_low = torch.tensor([0.1]).to(self.device)
        self.actor = Actor(self.n_in, self.n_mid, self.n_out, self.actor_hight, self.actor_low).to(self.device)
        self.critic = Critic(self.n_in, self.n_mid, self.n_out).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.global_brain = Brain(self.actor, self.critic, self.discriminator)

        #格納用の変数の生成
        self.obs_shape = [100, 100, 1]
        # print("self.obs_shape : ", self.obs_shape)
        self.current_obs = torch.zeros(NUM_PROCESSES, self.obs_shape[2], self.obs_shape[0], self.obs_shape[1]) # torch size ([16, 4])
        self.rollouts = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, self.obs_shape[2], self.obs_shape[0], self.obs_shape[1], action_num)
        self.episode_rewards = torch.zeros([NUM_PROCESSES, 1]) # 現在の施行の報酬を保持
        self.final_rewards = torch.zeros([NUM_PROCESSES, 1]) # 最後の施行の報酬を保持
        self.old_action_probs = torch.zeros([NUM_ADVANCED_STEP, NUM_PROCESSES, 1]) # ratioの計算のため
        self.obs_np = np.zeros([NUM_PROCESSES, self.obs_shape[2], self.obs_shape[0], self.obs_shape[1]]) 
        self.reward_np = np.zeros([NUM_PROCESSES, 1])
        self.done_np = np.zeros([NUM_PROCESSES, 1])
        self.each_step = np.zeros(NUM_PROCESSES) # 各環境のstep数を記録
        self.episode = 0 # 環境0の施行数

        # ポテンシャル法の部分
        self.Potential_avoid = Potential_avoid(delt=0.5, speed=0.5, weight_goal=30) # weight_obstはディープラーニングで計算

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
        state.transform = spawns[0]
        ego = self.sim.add_agent(self.env.str("LGSVL__VEHICLE_0", lgsvl.wise.DefaultAssets.seniorcar), lgsvl.AgentType.EGO, state)

        obstacle_foward = lgsvl.utils.transform_to_forward(spawns[0])
        obstacle_state = lgsvl.AgentState()
        obstacle_state.transform.position = spawns[0].position + 15.0 * obstacle_foward
        
        self.obstacle_ego = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, obstacle_state)
        self.obstacle_ego.transform

        self.lgsvl_objects = [
            self.obstacle_ego,
        ]

        ego.on_collision(self.on_collision)
        self.obstacle_ego.on_collision(self.on_collision)

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
        
        running_count = 0
        markerArray = MarkerArray()
        while not self.simulation_stop_flag: # simulation_stop_flagがFalseのときは実行し続ける
            # print("Runing !!!! step : ", step)
            self.sim.run(0.05)
            if running_count % 10 == 0:
                self.lgsvl_object_publisher()
                ego_quaternion = quaternion.from_rotation_vector(np.array([ego.state.transform.rotation.x, ego.state.transform.rotation.y, ego.state.transform.rotation.z]))
                # print("ego_quaternion : ", ego_quaternion)
                # NPCの位置を出力
                marker = Marker()
                marker.header.frame_id = "/lgsvlmap"
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.pose.orientation.x = ego_quaternion.x
                marker.pose.orientation.y = ego_quaternion.y
                marker.pose.orientation.z = ego_quaternion.z
                marker.pose.orientation.w = ego_quaternion.w
                marker.pose.position.x = self.obstacle_ego.state.transform.position.z + self.map_offset[0]
                marker.pose.position.y = -self.obstacle_ego.state.transform.position.x + self.map_offset[1]
                marker.pose.position.z = self.obstacle_ego.state.transform.position.y + self.map_offset[2]

                markerArray.markers.append(marker)
                id = 0
                for m in markerArray.markers:
                    m.id = id
                    id += 1
                self.npc_marker_publisher.publish(markerArray)

                # print("npc_state", type(ego))

            running_count += 1
        # print("SIMULATOR is reseted !!!")
        sys.exit()
    def on_collision(self, agent1, agent2, contact):
        name1 = self.objects[agent1]
        name2 = self.objects[agent2] if agent2 is not None else "OBSTACLE"
        print("{} collided with {} at {}".format(name1, name2, contact))
        self.on_collision_flag = True

    def check_error(self):
        current_pose = np.array([self.current_pose[0], 
                          self.current_pose[1], 
                          0])
        waypoint_pose = self.waypoint[self.closest_waypoint, 0:3]

        error_dist = np.sqrt(np.sum(np.square(waypoint_pose - current_pose)))

        done = False
        if error_dist > 1.0:
            done = True
        if self.closest_waypoint >= (len(self.waypoint) - 1 - 15):
            done = True
        
        return error_dist, 0, done, error_dist

    def env_feedback(self, actions):
        """------------経路をpublish------------"""
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
        velocity = self.waypoints[0, 4]
        change_flag = self.waypoints[0, 5]
        goal_flag, output_route = self.Potential_avoid.calculation(vehicle_position, goal_position, actions, self.gridmap, yaw, velocity, change_flag)
        # print("output_route : ", output_route.shape)
        # if goal_flag == True:
        result_route = output_route
        if goal_flag:
            # closest_waypointを探索
            error2waypoint = np.sum(np.abs(output_route[:, :2] - self.current_pose), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
            closest_waypoint = error2waypoint.argmin()
            publish_route = result_route
            multiarray = Float32MultiArray()
            multiarray.layout.dim.append(MultiArrayDimension())
            multiarray.layout.dim.append(MultiArrayDimension())
            multiarray.layout.dim[0].label = "height"
            multiarray.layout.dim[1].label = "width"
            multiarray.layout.dim[0].size = publish_route.shape[0]
            multiarray.layout.dim[1].size = publish_route.shape[1]
            multiarray.layout.dim[0].stride = publish_route.shape[0] * publish_route.shape[1]
            multiarray.layout.dim[1].stride = publish_route.shape[1]
            multiarray.data = publish_route.reshape(1, -1)[0].tolist()
            self.base_waypoint_publisher.publish(multiarray)

            self.waypoints = result_route
        time.sleep(0.2) # 0.2秒後の情報を得るために1秒のインターバル

        "-----------GANの識別器によって人っぽいか人っぽくないかを判別-------------"
        # 経路の正規化(経路waypointsをgridmap)にまとめる)
        generated_route_grid_value = np.zeros((self.grid_height, self.grid_width, 1))
        expert_route_grid_value = np.zeros((self.grid_height, self.grid_width, 1))
        route_grid_xy = np.array(np.meshgrid(np.linspace(-10, 10, num=100), np.linspace(0, 20, num=100))).transpose(1, 2, 0)
        # print("route_grid_xy : ", route_grid_xy.shape)


        # 生成されたresult_routeをgridmapにする.
        for route in result_route:
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
        for route in self.expert_waypoints:
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
        reward = 0
        done = False
        if goal_flag is not True: # ゴールできないような経路を作成してしまった場合
            done = True
            reward = -1
        if dist_vehicle2goal > 1.0: # ゴールに到達できなかった場合
            reward = -1
        if np.round(discriminator_output) == 1: # 識別器によって、生成されたrouteが人っぽいと判断された場合
            reward = 1

        return reward, done

    def init_environment(self):
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
        self.path_record = pd.DataFrame({"current_pose_x" : [self.current_pose[0]], "current_pose_y" : [self.current_pose[1]], "current_pose_z" : [0], 
                                    "closest_waypoint_x" : [self.waypoints[self.closest_waypoint][0]], "closest_waypoint_y" : [self.waypoints[self.closest_waypoint][1]], "closest_waypoint_z" : [self.waypoints[self.closest_waypoint][2]],
                                    "lookahead_distance" : [0], "error" : [0]})
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

        with open('data_{}/episode_mean10_{}.csv'.format(now_time, now_time), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["eisode", "finished_step", "10_step_meaning"])

        self.pandas_init()

        self.init_environment()

        # print("self.gridmap_object_value : ", self.gridmap_object_value.shape)
        self.rollouts.observations[0] = torch.from_numpy(self.gridmap_object_value.transpose(2, 0, 1)).to(self.device)
        episode = 0
        frame = 0

        for j in tqdm(range(NUM_EPISODES * NUM_PROCESSES)):
            state =  self.gridmap_object_value.transpose(2, 0, 1)#誤差を状態として学習させたい

            state = torch.from_numpy(state).type(torch.FloatTensor).to(self.device)
            state = torch.unsqueeze(state, 0)
            # print("state : ", state.shape)

            tmp_action_probs = torch.zeros([NUM_ADVANCED_STEP, NUM_PROCESSES, self.n_out])
            
            for step in range(NUM_ADVANCED_STEP):
                with torch.no_grad():
                    action, tmp_action_probs[step] = self.actor.act(self.rollouts.observations[step])
                actions = action.squeeze().cpu().numpy()
                print("actions : ", actions)
                
                for i in range(NUM_PROCESSES):
                    self.reward_np[i], self.done_np[i] = self.env_feedback(actions=actions) # errorを次のobsercation_next(次の状態)として扱う
                    self.obs_np[i] = self.gridmap_object_value.transpose(2, 0, 1) # 次の状態を格納
                    
                    # self.path_record = self.path_record.append({"current_pose_x" : self.current_pose.pose.position.x,
                    #                     "current_pose_y" : self.current_pose.pose.position.y,
                    #                     "current_pose_z" : self.current_pose.pose.position.z,
                    #                     "closest_waypoint_x" : self.waypoint[self.closest_waypoint][0],
                    #                     "closest_waypoint_y" : self.waypoint[self.closest_waypoint][1],
                    #                     "closest_waypoint_z" : self.waypoint[self.closest_waypoint][2],
                    #                     "lookahead_distance" : self.lookahead_dist.data,
                    #                     }, ignore_index=True)

                    if self.done_np[i]: # simulationが止まっていなかったらFalse, 終了するならTrue
                        state_next = None

                        episode_10_array[episode % 10] = frame
                        
                        if episode % 10 == 0: # 10episodeごとにウェイトを保存
                            torch.save(self.global_brain.actor, "./data_{}/weight/episode_{}_finish.pth".format(now_time, episode))

                        if i == 0: # 分散処理の最初の項が終了した場合、結果を記録
                            self.finish_environment()
                            print("%d Episode: Finished after %d frame : 10試行の平均frame数 = %.1lf" %(episode, frame, episode_10_array.mean()))
                            with open('data_{}/episode_mean10_{}.csv'.format(now_time, now_time).format(), 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow([episode, frame, episode_10_array.mean()])
                            episode += 1
                            frame = 0

                        if self.reward_np[i] == -1: # 途中で上手く走行できなかったら罰則として-1
                            print("[失敗]報酬 : - 1")
                            complete_episodes[episode % 10] = 0 # 連続成功記録をリセット
                            self.path_record.to_csv("./data_{}/learning_log_{}_ep{}_failure.csv".format(now_time, now_time, episode))
                        
                        elif self.reward_np[i] == 1: # 何事もなく、上手く走行出来たら報酬として+1
                            print("[成功]報酬 : + 1")
                            complete_episodes[episode % 10] = 1 # 連続成功記録を+1
                            self.path_record.to_csv("./data_{}/learning_log_{}_ep{}_success.csv".format(now_time, now_time, episode))

                        self.pandas_init()
                        # done がTrueのとき、環境のリセット(分散学習のとき、env[i].reset()みたいなことをしないといけない. その場合、ワークステーション10台くらい必要)
                        self.finish_environment()
                        time.sleep(3)
                        self.init_environment()

                    else:
                        self.each_step[i] += 1
                    
                frame += 1

                # 報酬をtensorに変換し、施行の総報酬に足す
                reward = torch.from_numpy(self.reward_np).float()
                self.episode_rewards += reward

                # 各実行環境それぞれについて、doneならmaskは0に、継続中ならmaskは1にする
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in self.done_np])

                # 最後の試行の報酬を更新する
                self.final_rewards *= masks # done==0の場所のみ0になる

                # 継続中は0を足す、done時にはepisode_rewardsを足す
                self.final_rewards += (1 - masks) * self.episode_rewards

                # 試行の総報酬を更新する
                self.episode_rewards *= masks

                # 現在の状態をdone時には全部0にする
                self.current_obs *= masks

                # current_obsを更新
                obs = torch.from_numpy(self.obs_np).float()
                self.current_obs = obs

                # メモリ王ジェクトに今のstepのtransitionを挿入
                # print("self.current_obs : ", self.current_obs.shape)
                # print("action.data : ", action.data.shape)
                # print("reward : ", reward.shape)
                # print("masks : ", masks.shape)
                self.rollouts.insert(self.current_obs, action.data, reward, masks)

            # advancedのfor文　loop終了

            with torch.no_grad():
                next_value = self.critic.get_value(self.rollouts.observations[-1].to(self.device).detach())
            
            self.rollouts.compute_returns(next_value)

            if j == 0:
                self.global_brain.actorcritic_update(self.rollouts, self.old_action_probs.to(self.device), first_episode=True)
            else:
                self.global_brain.actorcritic_update(self.rollouts, self.old_action_probs.to(self.device), first_episode=False)
            
            self.old_action_probs = tmp_action_probs # 昔(old)のaction_probsをここで保存

            self.rollouts.after_update()

            if np.sum(complete_episodes) >= NUM_COMPLETE_EP:
                print("8/10回連続成功")
                
                torch.save(self.global_brain.actor_critic, "./data_{}/weight/episode_{}_finish.pth".format(now_time, episode))
                print("無事に終了")
                episode_final = True
                break
            
            
            





