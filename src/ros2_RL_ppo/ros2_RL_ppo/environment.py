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

from .brain import Net, Brain, RolloutStorage

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
NUM_ADVANCED_STEP = 5
NUM_COMPLETE_EP = 10

os.chdir("/home/chohome/Master_research/LGSVL/ros2_RL_ws/src/ros2_RL_ppo/ros2_RL_ppo")
print("current pose : ", os.getcwd())

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")
now_JST = datetime.datetime.now(JST)
now_time = now_JST.strftime("%Y%m%d%H%M%S")
os.makedirs("./data_{}/weight".format(now_time))

# Python program raising
# exceptions in a python
# thread
  
class Environment(Node):
    def __init__(self):
        super().__init__("rl_environment")
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

        self.current_pose_sub = self.create_subscription(PoseStamped, "current_pose", self.current_poseCallback, 1)
        self.imu_sub = self.create_subscription(Imu, "imu_raw",  self.imuCallback, 1)
        self.closest_waypoint_sub = self.create_subscription(Int32, "closest_waypoint", self.closestWaypointCallback, 1)

        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 1)
        self.lookahead_pub = self.create_publisher(Float32, "minimum_lookahead_distance", 1)

        # その他Flagや設定
        self.initialpose_flag = False # ここの値がTrueなら, initialposeによって自己位置が完了したことを示す。
        self.waypoint = pd.read_csv("/home/chohome/Master_research/LGSVL/route/LGSeocho_simpleroute0.5.csv", header=None, skiprows=1).to_numpy()
        # print("waypoint : \n", self.waypoint)
        # print("GPU is : ", torch.cuda.is_available())

        self.n_in = 1 # 状態
        self.n_out = 2 #行動
        self.n_mid = 32
        self.actor_critic = Net(self.n_in, self.n_mid, self.n_out)

        self.global_brain = Brain(self.actor_critic)

        #格納用の変数の生成
        self.obs_shape = self.n_in
        # print("self.obs_shape : ", self.obs_shape)
        self.current_obs = torch.zeros(NUM_PROCESSES, self.obs_shape) # torch size ([16, 4])
        self.rollouts = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, self.obs_shape)
        self.episode_rewards = torch.zeros([NUM_PROCESSES, 1]) # 現在の施行の報酬を保持
        self.final_rewards = torch.zeros([NUM_PROCESSES, 1]) # 最後の施行の報酬を保持
        self.old_action_probs = torch.zeros([NUM_ADVANCED_STEP, NUM_PROCESSES, 1]) # ratioの計算のため
        self.obs_np = np.zeros([NUM_PROCESSES, self.obs_shape]) 
        self.reward_np = np.zeros([NUM_PROCESSES, 1])
        self.done_np = np.zeros([NUM_PROCESSES, 1])
        self.each_step = np.zeros(NUM_PROCESSES) # 各環境のstep数を記録
        self.episode = 0 # 環境0の施行数

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
        if error_dist > 1.0:
            done = True
        if self.closest_waypoint >= (len(self.waypoint) - 1 - 15):
            done = True
        
        return error_dist, 0, done, error_dist

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

        complete_episodes = 0 # 連続で上手く走行した試行数
        self.closest_waypoint = 1
        episode_final = False # 最後の試行フラグ

        with open('data/episode_mean10_{}.csv'.format(now_time), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["eisode", "finished_step", "10_step_meaning"])

        self.pandas_init()

        self.init_environment()
        
        episode = 0
        frame = 0
        for j in range(NUM_EPISODES * NUM_PROCESSES):

            pose_state = self.current_pose.pose
            imu_state = self.imu.linear_acceleration # 要調整
                        
            state = np.array([self.check_error()[0]]) #誤差を状態として学習させたい

            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)


            for step in range(NUM_ADVANCED_STEP):
            # while self.closest_waypoint < (len(self.waypoint) - 1 - 5) and self.simulation_stop_flag == False: # 1エピソードのループ ※len(waypoint)はwaypointの総数だが, 0インデックスを考慮して -1 その次に 経路の終わりから3つ前までという意味で -3
                # print("state : ", state)
                with torch.no_grad():
                    action, self.old_action_probs[step] = self.actor_critic.act(self.rollouts.observations[step])
                actions = action.squeeze(1).numpy()
                
                # action(purepursuit or 無限遠点)をパブリッシュしてpure pursuit本体に送る
                self.lookahead_dist = Float32()
                if action == 0:
                    self.lookahead_dist.data = 0.5
                else:
                    self.lookahead_dist.data = 5.0
                self.lookahead_pub.publish(self.lookahead_dist)

                time.sleep(1) # １秒後の情報を得るために1秒のインターバル
                
                for i in range(NUM_PROCESSES):
                    self.obs_np[i], self.reward_np[i], self.done_np[i], error_distance = self.check_error() # errorを次のobsercation_next(次の状態)として扱う
                    # print("error : ", self.obs_np[i])
                    self.path_record = self.path_record.append({"current_pose_x" : self.current_pose.pose.position.x,
                                        "current_pose_y" : self.current_pose.pose.position.y,
                                        "current_pose_z" : self.current_pose.pose.position.z,
                                        "closest_waypoint_x" : self.waypoint[self.closest_waypoint][0],
                                        "closest_waypoint_y" : self.waypoint[self.closest_waypoint][1],
                                        "closest_waypoint_z" : self.waypoint[self.closest_waypoint][2],
                                        "lookahead_distance" : self.lookahead_dist.data,
                                        "error" : error_distance
                                        }, ignore_index=True)
                    
                    # print("path_record : \n", self.path_record)

                    if self.done_np[i]: # simulationが止まっていなかったらFalse, 終了するならTrue
                        state_next = None

                        episode_10_array[episode % 10] = frame
                        
                        if episode % 10 == 0: # 10episodeごとにウェイトを保存
                            torch.save(self.global_brain.actor_critic, "./data_{}/weight/episode_{}_finish.pth".format(now_time, episode))

                        if i == 0: # 分散処理の最初の項が終了した場合、結果を記録
                            self.finish_environment()
                            print("%d Episode: Finished after %d frame : 10試行の平均frame数 = %.1lf" %(episode, frame, episode_10_array.mean()))
                            with open('data/episode_mean10_{}.csv'.format(now_time).format(), 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow([episode, frame, episode_10_array.mean()])
                            episode += 1
                            frame = 0

                        if self.closest_waypoint < (len(self.waypoint) - 1 - 15): # 途中で上手く走行できなかったら罰則として-1
                            print("[失敗]報酬 : - 1")
                            self.reward_np[i] = torch.FloatTensor([-1.0])
                            complete_episodes = 0 # 連続成功記録をリセット
                            self.path_record.to_csv("./data_{}/learning_log_{}_ep{}_failure.csv".format(now_time, now_time, episode))
                        
                        else: # 何事もなく、上手く走行出来たら報酬として+1
                            print("[成功]報酬 : + 1")
                            self.reward_np[i] = torch.FloatTensor([1.0])
                            complete_episodes = complete_episodes + 1 # 連続成功記録を+1
                            self.path_record.to_csv("./data_{}/learning_log_{}_ep{}_success.csv".format(now_time, now_time, episode))

                        self.pandas_init()
                        # done がTrueのとき、環境のリセット(分散学習のとき、env[i].reset()みたいなことをしないといけない. その場合、ワークステーション10台くらい必要)
                        self.finish_environment()
                        time.sleep(3)
                        self.init_environment()

                    else:
                        self.reward_np[i] = torch.FloatTensor([0.0]) # 普段は報酬0
                        # state_next = np.array([observation_next]) # 誤差をnp.arrayに変更
                        # state_next = torch.from_numpy(state_next).type(torch.FloatTensor)

                        # state_next = torch.unsqueeze(state_next, 0)
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
                self.rollouts.insert(self.current_obs, action.data, reward, masks)

            # advancedのfor文　loop終了

            with torch.no_grad():
                next_value = self.actor_critic.get_value(self.rollouts.observations[-1].detach())
            
            self.rollouts.compute_returns(next_value)

            if j == 0:
                self.global_brain.update(self.rollouts, self.old_action_probs, first_episode=True)
            else:
                self.global_brain.update(self.rollouts, self.old_action_probs, first_episode=False)

            self.rollouts.after_update()

            if complete_episodes >= NUM_COMPLETE_EP:
                print("10回連続成功")
                
                torch.save(self.global_brain.actor_critic, "./data_{}/weight/episode_{}_finish.pth".format(now_time, episode))
                print("無事に終了")
                episode_final = True
                break

            # frame += 1
            
            
            





