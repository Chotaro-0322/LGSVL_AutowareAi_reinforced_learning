# system packages
import os
import sys
import time

# ROS packages
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose
from rclpy.clock import Clock


# python packages
import open3d as o3d
import numpy as np
import pandas as pd

# Utils
from .ros2_numpy.ros2_numpy.point_cloud2 import pointcloud2_to_array
from .ros2_numpy.ros2_numpy.occupancy_grid import occupancygrid_to_numpy
from .avoid import Potential_avoid

np.set_printoptions(threshold=10000)

class Astar_avoid(Node):
    def __init__(self):
        super().__init__("astar_avoid")

        self.Potential_avoid = Potential_avoid(delt=0.5, speed=0.5, weight_obst=0.1, weight_goal=10)

        self.pcd_as_numpy_array = np.zeros((0))
        self.gridmap_as_numpy_array = np.zeros((0))
        self.current_pose = np.zeros([0, 0])
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
        self.costmap_subscriber = self.create_subscription(OccupancyGrid, "/semantics/costmap_generator/occupancy_grid", self.costmapCallback, 1)
        self.ndt_pose_subscriber = self.create_subscription(PoseStamped, "ndt_pose", self.ndtPoseCallback, 1)
        
        self.waypoint_publisher = self.create_publisher(Float32MultiArray, "route_waypoints_multiarray", 1)
        self.closest_waypoint_publisher = self.create_publisher(Int32, "closest_waypoint", 1)

    def pointcloudCallback(self, msg):
        self.pcd_as_numpy_array = pointcloud2_to_array(msg)

    def costmapCallback(self, msg):
        self.grid_resolution = msg.info.resolution
        self.grid_width = msg.info.width
        self.grid_height = msg.info.height
        self.grid_position = msg.info.origin
        self.gridmap_as_numpy_array = occupancygrid_to_numpy(msg)[:, :, np.newaxis] # あとで座標情報とくっつけるために次元を増やしておく

        self.grid_coordinate_array = np.zeros((self.grid_height, self.grid_width, 2))
        for i in range(self.grid_width): # /mapの座標を設定 : x軸
            self.grid_coordinate_array[:, i, 1] = self.grid_position.position.x + i * self.grid_resolution
        for i in range(self.grid_height): # /mapの座標を設定 : y軸
            self.grid_coordinate_array[i, :, 0] = self.grid_position.position.y + i * self.grid_resolution

        self.gridmap = np.block([self.grid_coordinate_array, self.gridmap_as_numpy_array])

    def ndtPoseCallback(self, msg):
        self.current_pose = np.array([msg.pose.position.x, msg.pose.position.y])

    def run(self):
        waypoints = pd.read_csv("/home/itolab-chotaro/HDD/Master_research/LGSVL/route/LGSeocho_toavoid0.5.csv", header=None, skiprows=1).to_numpy()
        # -------------------------------
        #       ここに一度ウェイポイントを出力する項目を作成及びpublishする
        #--------------------------------
        multiarray = Float32MultiArray()
        multiarray.layout.dim.append(MultiArrayDimension())
        multiarray.layout.dim.append(MultiArrayDimension())
        multiarray.layout.dim[0].label = "height"
        multiarray.layout.dim[1].label = "width"
        multiarray.layout.dim[0].size = waypoints.shape[0]
        multiarray.layout.dim[1].size = waypoints.shape[1]
        multiarray.layout.dim[0].stride = waypoints.shape[0] * waypoints.shape[1]
        multiarray.layout.dim[1].stride = waypoints.shape[1]
        multiarray.data = waypoints.reshape(1, -1)[0].tolist()

        self.waypoint_publisher.publish(multiarray)
        time.sleep(0.5)

        while True:
            # gridmapが来るまで待機
            while True: 
                if len(list(self.gridmap)) != 0 and len(self.current_pose) != 0: # Gridmapがsubscribeされたら
                    break
                else:
                    time.sleep(0.2)
            
            # closest_waypointを探索
            error2waypoint = np.sum(np.abs(waypoints[:, :2] - self.current_pose), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
            # print("error2waypoint : ", error2waypoint)
            closest_waypoint = error2waypoint.argmin()
            print("closest_waypoint : ", closest_waypoint)
            closest_waypoint_msg = Int32()
            closest_waypoint_msg.data = int(closest_waypoint)
            self.closest_waypoint_publisher.publish(closest_waypoint_msg)
            # print("closest_waypoint : ", closest_waypoint)

            # waypointと照らし合わせて、waypointの近くに障害物があったら self.object_bool = Trueにする.
            obstacle_value = 0
            for i in range(self.length_judge_obstacle):
                '''-----waypointのx, y の値をGridmapの座標から引いて差が一番小さいところを探すことによって, waypointがどのgridマスに近いかを計算-----'''
                # 計算しやすくするために各ウェイポイントの座標をgridサイズの縦 x 横で埋める
                grid_from_waypoint = np.stack([np.full((self.grid_height, self.grid_width), waypoints[closest_waypoint + i, 1])
                                              ,np.full((self.grid_height, self.grid_width), waypoints[closest_waypoint + i, 0])], -1)
                
                # gridmapとgrid_fram_waypointの差を計算
                error_grid_space = np.sum(np.abs(self.gridmap[:, :, :2] - grid_from_waypoint), axis=2)
                # 計算された差から, 一番値が近いグリッドを計算
                nearest_grid_space = [np.argmin(error_grid_space, axis=0)[0], np.argmin(error_grid_space, axis=1)[0]]
                # print("grid height : ", self.grid_height , " , grid width" , self.grid_width, "\n")
                # print("nearest_grid_space : ", nearest_grid_space)

                '''----計算されたグリッドマップ周辺に障害物がないか計算----'''
                # マス周辺5マス分の障害物の値を得る(obstacle_value > 0のときに前方に障害物あり)
                obstacle_value += np.sum(self.gridmap[nearest_grid_space[0]-10 : nearest_grid_space[0]+10, nearest_grid_space[1]-10 : nearest_grid_space[1]+10, 2])
                print("waypoints_number : ", self.num_closest_waypoint + i, "| 障害物の値 : ", obstacle_value)

                if i == 0: # closest_waypointのときは自車のいるgridマスとして保存しておく(あとで回避時のスタート地点として使用)
                    print("vehicle waypoint : ", i)
                    self.vehicle_closest_waypoint = closest_waypoint + i
                    self.vehicle_grid = self.gridmap[nearest_grid_space[0], nearest_grid_space[1]][:2]
                elif i == self.length_judge_obstacle - 1: # 探索waypointsの最後の点を避けるときのゴール地点に設定しておく
                    print("goal waypoint : ", i)
                    self.goal_closest_waypoint = closest_waypoint + i
                    self.goal_grid = self.gridmap[nearest_grid_space[0], nearest_grid_space[1]][:2]

            "----ポテンシャル法による回避----"
            if obstacle_value > 0: # 前方に障害物があったら
                # astarの起動!!!!
                print("waypoint shape : ", waypoints.shape)
                
                vehicle_position = np.array(self.vehicle_grid)
                goal_position = np.array([waypoints[-1, 1], waypoints[-1, 0]])
                yaw = waypoints[self.vehicle_closest_waypoint, 3]
                velocity = waypoints[0, 4]
                change_flag = waypoints[0, 5]
                # print("vehicle_start point : ", vehicle_position)
                # print("goal point : ", goal_position)
                goal_flag, output_route = self.Potential_avoid.calculation(vehicle_position, goal_position, self.gridmap, yaw, velocity, change_flag)
                
                # result_route = np.concatenate([waypoints[:self.vehicle_closest_waypoint], output_route, waypoints[self.goal_closest_waypoint:]])
                # result_route = np.concatenate([waypoints[:self.vehicle_closest_waypoint], output_route])
                result_route = output_route
                np.savetxt('./base_waypoints.csv', waypoints, delimiter=",")
                np.savetxt('./avoid_waypoints.csv', result_route, delimiter=",")
                # input()
                print("start_waypoint : ", self.vehicle_closest_waypoint)
                print("goal waypoint : ", self.goal_closest_waypoint)
                print("all_of_waypoint : ", waypoints.shape[0])
                print("result waypoint shape : ", result_route.shape)
                if goal_flag == True:
                    multiarray = Float32MultiArray()
                    multiarray.layout.dim.append(MultiArrayDimension())
                    multiarray.layout.dim.append(MultiArrayDimension())
                    multiarray.layout.dim[0].label = "height"
                    multiarray.layout.dim[1].label = "width"
                    multiarray.layout.dim[0].size = result_route.shape[0]
                    multiarray.layout.dim[1].size = result_route.shape[1]
                    multiarray.layout.dim[0].stride = result_route.shape[0] * result_route.shape[1]
                    multiarray.layout.dim[1].stride = result_route.shape[1]
                    multiarray.data = result_route.reshape(1, -1)[0].tolist()
            #     else:
            #         multiarray = Float32MultiArray()
            #         multiarray.layout.dim.append(MultiArrayDimension())
            #         multiarray.layout.dim.append(MultiArrayDimension())
            #         multiarray.layout.dim[0].label = "height"
            #         multiarray.layout.dim[1].label = "width"
            #         multiarray.layout.dim[0].size = waypoints.shape[0]
            #         multiarray.layout.dim[1].size = waypoints.shape[1]
            #         multiarray.layout.dim[0].stride = waypoints.shape[0] * waypoints.shape[1]
            #         multiarray.layout.dim[1].stride = waypoints.shape[1]
            #         multiarray.data = waypoints.reshape(1, -1)[0].tolist()

                    self.waypoint_publisher.publish(multiarray)
                    time.sleep(2.0)
                    

            #     waypoints = result_route
            #     time.sleep(0.5)

            # 前方に障害物がなかったら、そのままウェイポイントを出力(Vehicle_cmd_filter.pyに送られ、そこでfinal_waypointsとしてpublishされる。)
                    # なぜfinal_waypointsとして直接publishしないのかというと、ROS1_bridgeはautoware_msgにデフォルトで対応していないので、ros2内でautoware.aiのautoware_msgs.msg Laneが使えないことが理由
                # base_waypointsの出力をする
                


