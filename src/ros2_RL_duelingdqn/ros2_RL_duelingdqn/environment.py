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

from .agent import Agent

import numpy as np
import torch
import json
import time
import threading
import ctypes
import pandas as pd
import csv
import datetime

MAX_STEPS = 1000
NUM_EPISODES = 1000

os.chdir("/home/chohome/Master_research/LGSVL/ros2_RL_ws/src/ros2_RL_duelingdqn/ros2_RL_duelingdqn")
print("current pose : ", os.getcwd())

# Python program raising
# exceptions in a python
# thread
 
import threading
import ctypes
import time
  
class Environment(Node):
    def __init__(self):
        super().__init__("rl_environment")
        # 環境の準備
        self.env = Env() 
        self.sim = lgsvl.Simulator(self.env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host), self.env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
        print("str : ", self.env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host))
        print("int : ", self.env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
        self.simulation_stop_flag = False

        self.num_states = 1 # ニューラルネットワークの入力数
        self.num_actions = 2 # purepursuit 0.5mと2m
        self.agent = Agent(self.num_states, self.num_actions) # Agentを作成(self.num_statesは要調整)

        # ROS setting
        self.current_pose = PoseStamped()
        self.imu = Imu()
        self.closest_waypoint = Int32()

        self.current_pose_sub = self.create_subscription(PoseStamped, "current_pose", self.current_poseCallback, 1)
        self.imu_sub = self.create_subscription(Imu, "imu_raw",  self.imuCallback, 1)
        self.closest_waypoint_sub = self.create_subscription(Int32, "closest_waypoint", self.closestWaypointCallback, 1)

        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 1)
        self.lookahead_pub = self.create_publisher(Float32, "minimum_lookahead_distance", 1)

        # その他Flagや設定
        self.initialpose_flag = False # ここの値がTrueなら, initialposeによって自己位置が完了したことを示す。
        self.waypoint = pd.read_csv("/home/chohome/Master_research/LGSVL/route/LGSeocho_simpleroute_bitshort.csv", header=None, skiprows=1).to_numpy()
        self.now_time = datetime.datetime.now().time()
        # print("waypoint : \n", self.waypoint)
        # print("GPU is : ", torch.cuda.is_available())

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

        # controllables object setting
        with open("./warehouse_map_ver2.json") as f:
            jsn = json.load(f)
            # print("jsn : ", jsn)
            for controllables in jsn["controllables"]:
                state = lgsvl.ObjectState()
                state.transform.position = lgsvl.Vector(controllables["transform"]["position"]["x"],
                                                        controllables["transform"]["position"]["y"],
                                                        controllables["transform"]["position"]["z"],)
                state.transform.rotation = lgsvl.Vector(controllables["transform"]["rotation"]["x"],
                                                        controllables["transform"]["rotation"]["y"],
                                                        controllables["transform"]["rotation"]["z"],)
                state.velocity = lgsvl.Vector(0, 0, 0)
                state.angular_velocity = lgsvl.Vector(0, 0, 0)

                cone = self.sim.controllable_add(controllables["name"], state)
            
            for i, agents in enumerate(jsn["agents"]):
                if i > 0:
                    state = lgsvl.ObjectState()
                    state.transform.position = lgsvl.Vector(agents["transform"]["position"]["x"],
                                                            agents["transform"]["position"]["y"],
                                                            agents["transform"]["position"]["z"],)
                    state.transform.rotation = lgsvl.Vector(agents["transform"]["rotation"]["x"],
                                                            agents["transform"]["rotation"]["y"],
                                                            agents["transform"]["rotation"]["z"],)
                    state.velocity = lgsvl.Vector(0, 0, 0)
                    state.angular_velocity = lgsvl.Vector(0, 0, 0)
                    # print("car name ]: ", agents["variant"])
                    npc = self.sim.add_agent(agents["variant"], lgsvl.AgentType.NPC, state=state)

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

    def check_error(self):
        current_pose = np.array([self.current_pose.pose.position.x, 
                          self.current_pose.pose.position.y, 
                          self.current_pose.pose.position.z])
        waypoint_pose = self.waypoint[self.closest_waypoint, 0:3]

        error_dist = np.sqrt(np.sum(np.square(waypoint_pose - current_pose)))

        # print("current_pose : \n", current_pose)
        # print("waypoint_pose : \n", waypoint_pose)
        # print("pose_difference : ", error_dist)

        done = False
        if error_dist > 1.3:
            done = True
        if self.closest_waypoint >= (len(self.waypoint) - 1 - 3):
            done = True
        
        return current_pose, done, error_dist

    def run(self):
        episode_10_array = np.zeros(10) # 10試行分の"経路から外れない", "IMUによる蛇行がしない"step数を格納し, 平均ステップ数を出力に利用

        complete_episodes = 0 # 連続で上手く走行した試行数
        self.closest_waypoint = 1
        episode_final = False # 最後の試行フラグ

        with open('data/episode_mean10_{}.csv'.format(self.now_time), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["eisode", "finished_step", "10_step_meaning"])

        for episode in range(NUM_EPISODES):
            self.simulation_stop_flag = False
            env_thread = threading.Thread(target = self.environment_build, name = "LGSVL_Environment")
            env_thread.start()

            self.initialpose_flag = False
            # current_poseが送られてきて自己位置推定が完了したと思われたら self.initalpose_flagがcurrent_pose_callbackの中でTrueになり, whileから抜ける
            while not self.initialpose_flag:
                # print("current_pose no yet ...!")
                time.sleep(0.1)
            
            pose_state = self.current_pose.pose
            imu_state = self.imu.linear_acceleration # 要調整
            # print("Imu states : ", imu_state)

            # print("state : ", np.array([self.check_error()[2]]))
            state = np.array([self.check_error()[2]]) #誤差を状態として学習させたい


            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            path_record = pd.DataFrame({"current_pose_x" : [self.current_pose.pose.position.x], "current_pose_y" : [self.current_pose.pose.position.y], "current_pose_z" : [self.current_pose.pose.position.z], 
                                        "closest_waypoint_x" : [self.waypoint[self.closest_waypoint][0]], "closest_waypoint_y" : [self.waypoint[self.closest_waypoint][1]], "closest_waypoint_z" : [self.waypoint[self.closest_waypoint][2]],
                                        "lookahead_distance" : [0], "error" : [0]})
            # print("Init path_record : \n", path_record)

            step = 0
            while self.closest_waypoint < (len(self.waypoint) - 1 - 3) and self.simulation_stop_flag == False: # 1エピソードのループ ※len(waypoint)はwaypointの総数だが, 0インデックスを考慮して -1 その次に 経路の終わりから3つ前までという意味で -3
                # print("state : ", state)
                action = self.agent.get_action(state, episode)
                
                # action(purepursuit or 無限遠点)をパブリッシュしてpure pursuit本体に送る
                self.lookahead_dist = Float32()
                if action == 0:
                    self.lookahead_dist.data = 0.5
                else:
                    self.lookahead_dist.data = 5.0
                self.lookahead_pub.publish(self.lookahead_dist)

                time.sleep(1)

                next_pose, done, observation_next = self.check_error() # errorを次のobsercation_next(次の状態)として扱う

                path_record = path_record.append({"current_pose_x" : self.current_pose.pose.position.x,
                                    "current_pose_y" : self.current_pose.pose.position.y,
                                    "current_pose_z" : self.current_pose.pose.position.z,
                                    "closest_waypoint_x" : self.waypoint[self.closest_waypoint][0],
                                    "closest_waypoint_y" : self.waypoint[self.closest_waypoint][1],
                                    "closest_waypoint_z" : self.waypoint[self.closest_waypoint][2],
                                    "lookahead_distance" : self.lookahead_dist.data,
                                    "error" : observation_next
                                    }, ignore_index=True)

                if done: # simulationが止まっていなかったらFalse, 終了するならTrue
                    state_next = None

                    episode_10_array[episode % 10] = step

                    if self.closest_waypoint < (len(self.waypoint) - 1 - 3): # 途中で上手く走行できなかったら罰則として-1
                        print("[失敗]報酬 : - 1")
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0 # 連続成功記録をリセット
                        path_record.to_csv("./data/learning_log_{}_ep{}_failure.csv".format(self.now_time, episode))
                    
                    else: # 何事もなく、上手く走行出来たら報酬として+1
                        print("[成功]報酬 : + 1")
                        reward = torch.FloatTensor([1.0])
                        complete_episodes = complete_episodes + 1 # 連続成功記録を+1
                        path_record.to_csv("./data/learning_log_{}_ep{}_success.csv".format(self.now_time, episode))

                else:
                    reward = torch.FloatTensor([0.0]) # 普段は報酬0
                    state_next = np.array([observation_next]) # 誤差をnp.arrayに変更
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)

                    state_next = torch.unsqueeze(state_next, 0)

                # メモリに経験を追加
                self.agent.memorize(state, action, state_next, reward)

                # TD誤差メモリにTD誤差を追加
                self.agent.memorize_td_error(0)

                # Experience ReplayでQ関数を更新
                self.agent.update_q_function(episode)

                state = state_next
                # print("self.simulation_stop_flag : ", self.simulation_stop_flag)
                
                step += 1
                

                # 終了時の処理
                if done:
                    self.simulation_stop_flag = True # シミュレータを終了させる
                    env_thread.join()
                    print("%d Episode: Finished after %d steps : 10試行の平均step数 = %.1lf" %(episode, step, episode_10_array.mean()))
                    with open('data/episode_mean10_{}.csv'.format(self.now_time).format(), 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([episode, step, episode_10_array.mean()])
                    
                    self.agent.update_td_error_memory()

                    # ターゲットネットワークを更新
                    if (episode % 2 == 0):
                        self.agent.update_target_q_function()
                    
                    if episode % 5 == 0:
                        torch.save(self.agent.brain.main_q_network, "./data/weight/episode_{}.pth".format(episode))
                    
                    time.sleep(0.5)
                    break

                
                
            if episode_final is True:
                torch.save(self.agent.brain.model, "./data/weight/episode_{}_finish.pth".format(episode))
                print("無事に終了")
                break
            
            # 10連続で走行し終えたら成功
            if complete_episodes >= 10:
                print("10回連続成功")
                episode_final = True
            
            





