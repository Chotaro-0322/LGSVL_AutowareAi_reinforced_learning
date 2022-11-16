# system packages
import os
import sys
import time

# ROS packages
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose

# python packages
import open3d as o3d
import numpy as np
import pandas as pd

# Utils
from .ros2_numpy.ros2_numpy.point_cloud2 import pointcloud2_to_array
from .ros2_numpy.ros2_numpy.occupancy_grid import occupancygrid_to_numpy

np.set_printoptions(threshold=10000)

class Astar_avoid(Node):
    def __init__(self):
        super().__init__("astar_avoid")

        self.pcd_as_numpy_array = np.zeros((0))
        self.gridmap_as_numpy_array = np.zeros((0))
        # print(self.gridmap_as_numpy_array)
        self.current_pose = PoseStamped()
        self.num_closest_waypoint = 0
        self.gridmap = np.zeros((0))

        self.grid_resolution = 0
        self.grid_width = 0
        self.grid_height = 0
        self.grid_position = Pose()

        self.length_judge_obstacle = 10 # この数字分, 先のwaypointを使って障害物を判断

        self.pcd_subscriber = self.create_subscription(PointCloud2, "clipped_cloud", self.pointcloudCallback, 1)
        self.costmap_subscriber = self.create_subscription(OccupancyGrid, "/semantics/costmap_generator/occupancy_grid", self.costmapCallback, 1)
        self.current_pose_subscriber = self.create_subscription(PoseStamped, "current_pose", self.currentposeCallback, 1)
        self.closest_waypoint_subscriber = self.create_subscription(Int32, "closest_waypoint", self.closestWaypointCallback, 1)

    def pointcloudCallback(self, msg):
        self.pcd_as_numpy_array = pointcloud2_to_array(msg)
        # print("pcs_as_numpy_array")

    def costmapCallback(self, msg):
        self.grid_resolution = msg.info.resolution
        self.grid_width = msg.info.width
        self.grid_height = msg.info.height
        self.grid_position = msg.info.origin
        self.gridmap_as_numpy_array = occupancygrid_to_numpy(msg)[:, :, np.newaxis] # あとで座標情報とくっつけるために次元を増やしておく
        print("gridmap is called")
        # print("grid origin : \n", self.grid_position)

        self.grid_coordinate_array = np.zeros((self.grid_height, self.grid_width, 2))
        for i in range(self.grid_width): # /mapの座標を設定 : x軸
            self.grid_coordinate_array[:, i, 1] = self.grid_position.position.x + i * self.grid_resolution
        for i in range(self.grid_height): # /mapの座標を設定 : y軸
            self.grid_coordinate_array[i, :, 0] = self.grid_position.position.y + i * self.grid_resolution
        

        # print("self.gridmap_numpy : ", self.gridmap_as_numpy_array)
        print("self.gridmap_coord : ", self.grid_coordinate_array.shape)

        self.gridmap = np.block([self.grid_coordinate_array, self.gridmap_as_numpy_array])
        # print("self.gridmap : \n", self.gridmap[:, :, 2])

    def currentposeCallback(self, msg):
        self.current_pose = msg
        # print("currentpose is called")

    def closestWaypointCallback(self, msg):
        self.num_closest_waypoint = msg.data
        # print("closest waypoint is called")

    def run(self):
        base_waypoints = pd.read_csv("/home/chohome/Master_research/LGSVL/route/LGSeocho_toavoid0.5.csv", header=None, skiprows=1).to_numpy()

        # -------------------------------
        #       ここに一度ウェイポイントを出力する項目を作成及びpublishする
        #--------------------------------
        # print("base_waypoints : \n", base_waypoints)
        while True:
            # gridmapが来るまで待機
            while True: 
                if len(list(self.gridmap)) != 0: # Gridmapがsubscribeされたら
                    break
                else:
                    time.sleep(0.2)

            # waypointと照らし合わせて、waypointの近くに障害物があったら self.object_bool = Trueにする.
            for i in range(self.length_judge_obstacle):
                '''-----waypointのx, y の値をGridmapの座標から引いて差が一番小さいところを探すことによって, waypointがどのgridマスに近いかを計算-----'''
                # 計算しやすくするために各ウェイポイントの座標をgridサイズの縦 x 横で埋める
                grid_from_waypoint = np.stack([np.full((self.grid_height, self.grid_width), base_waypoints[self.num_closest_waypoint + i, 1])
                                              ,np.full((self.grid_height, self.grid_width), base_waypoints[self.num_closest_waypoint + i, 0])], -1)
                
                # print("expanded grid : \n", grid_from_waypoint.shape)
                # gridmapとgrid_fram_waypointの差を計算
                error_grid_space = np.sum(np.abs(self.gridmap[:, :, :2] - grid_from_waypoint), axis=2)
                # 計算された差から, 一番値が近い座標を計算
                nearest_grid_space = [np.argmin(error_grid_space, axis=0)[0], np.argmin(error_grid_space, axis=1)[0]]
                # print("error_grid_space : ", error_grid_space.shape)
                print("grid height : ", self.grid_height , " , grid width" , self.grid_width, "\n")
                print("nearest_grid_space : ", nearest_grid_space)

                '''----計算されたグリッドマップ周辺に障害物がないか計算----'''
                # マス周辺5マス分の障害物の値を得る(obstacle_value > 0のときに前方に障害物あり)
                obstacle_value = np.sum(self.gridmap[nearest_grid_space[0]-10 : nearest_grid_space[0]+10, nearest_grid_space[1]-10 : nearest_grid_space[1]+10, 2])
                # print(self.gridmap[nearest_grid_space[0]-20 : nearest_grid_space[0]+20, nearest_grid_space[1]-30 : nearest_grid_space[1]+30, 2])
                print("waypoints_number : ", self.num_closest_waypoint + i, "| 障害物の値 : ", obstacle_value)

                if obstacle_value > 0: # 前方に障害物があったら
                    # astarの起動!!!!

                else: # 前方に障害物がなかったら、そのままウェイポイントを出力
                    # base_waypointsの出力をする


