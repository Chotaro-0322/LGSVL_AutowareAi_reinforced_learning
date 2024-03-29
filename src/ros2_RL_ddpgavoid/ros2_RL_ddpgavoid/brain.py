import os
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple
import json

import numpy as np

BATCH_SIZE = 64
NUM_STACK_FRAME = 4
CAPACITY = 500
GAMMA = 0.99

TD_ERROR_EPSILON = 0.0001

NUM_PROCESSES = 1
NUM_ADVANCED_STEP = 50
value_loss_coef = 0.5
entropy_coef = 0.5
max_grad_norm = 0.5
config_clip = 0.2


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))

class ReplayMemory():
    def __init__(self, CAPACITY, json_file_dir):
        self.capacity = CAPACITY
        self.global_npy_list = []
        self.index = 0
        self.json_file_dir = json_file_dir

    def push(self, state, action, state_next, reward, done, json_now_time):
        if len(self.global_npy_list) < self.capacity:
            self.global_npy_list.append(None)
        
        state = state.cpu().detach().numpy()
        action = action
        state_next = state_next.cpu().detach().numpy()
        reward = reward.cpu().detach().numpy()
        Transition_dict = Transition(state, action, state_next, reward, done)._asdict()
        np.save(os.path.join(self.json_file_dir , f"dict_{str(json_now_time)}.npy"), Transition_dict)

        self.global_npy_list[self.index] = os.path.join(self.json_file_dir , f"dict_{str(json_now_time)}.npy")
        self.index = (self.index + 1) % self.capacity # 保存しているindexを1つずらす　→　1001 % self.capacity = 1となる

    def sample(self, batch_size):
        batch_npy_list = random.sample(self.global_npy_list, batch_size)
        # print("json_list : ", batch_npy_list)
        memory = []
        for npy in batch_npy_list:
            npy_object = np.load(npy, allow_pickle=True).item()
            state = torch.from_numpy(npy_object["state"]).clone()
            action = npy_object["action"]
            state_next = torch.from_numpy(npy_object["next_state"]).clone()
            reward = torch.from_numpy(npy_object["reward"]).clone()
            done = npy_object["done"]
            memory.append(Transition(state, action, state_next, reward, done))
        return memory
    
    def __len__(self):
        return len(self.global_npy_list)

class TDerrorMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, td_error):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity
    
    def __len__(self):
        return len(self.memory)
    
    def get_prioritized_indexes(self, batch_size):
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)

        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)
        return indexes

    def update_td_error(self, updated_td_errors):
        self.memory = updated_td_errors

class Actor(nn.Module):
    def __init__(self, n_in, n_mid, n_out, action_space_high, action_space_low):
        super(Actor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)

        # Actor
        self.fc3 = nn.Linear(n_mid, len(action_space_high)) # [壁x, 壁y, 車x, 車y, 人x, 人y, ゴール]など

        self.action_center = (action_space_high + action_space_low)/2
        self.action_scale = action_space_high - self.action_center
        self.action_range = action_space_high - action_space_low

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actor_output = F.tanh(self.fc3(x)) # -1〜1にまとめる
        return actor_output

    def act(self, x, episode):
        # std = 1 / (0.05 * episode + 1)
        std = 0.2
        # print("x : ", x.size())
        action = self(x)
        # print("action : ", action.size())
        action = action.detach()
        # print("output action : ", action)
        noise = torch.normal(mean=torch.zeros(1), std=std).to(self.device)
        env_action = torch.clip(action + noise, -1, 1)
        print("env_action : ", env_action)

        return env_action * self.action_scale + self.action_center , env_action

class Critic(nn.Module):
    def __init__(self, obs_shape1, obs_shape2, obs_shape3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 64)

        # Critic
        self.critic = nn.Linear(64, 1) # 価値V側
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, actions):
        x = torch.cat((x, actions), dim = 1)
        # print("action : x -> ", x.size())
        x_2 = x
        # print("x: shape : ", x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        critic_output1 = self.critic(x)

        x_2 = F.relu(self.fc1(x_2))
        x_2 = F.relu(self.fc2(x_2))
        critic_output2 = self.critic(x_2)
        
        return critic_output1, critic_output2

    def get_value(self, x):
        value = self(x)
        return value

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, 2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, 2, stride=2)
        self.conv2d_3 = nn.Conv2d(64, 128, 2, stride=2)
        self.conv2d_4 = nn.Conv2d(128, 256, 2, stride=2)
        # self.conv2d_5 = nn.Conv2d(256, 1, 1, stride=1) # ボルトネック層

        self.batch_norm_1 = nn.InstanceNorm2d(32)
        self.batch_norm_2 = nn.InstanceNorm2d(64)
        self.batch_norm_3 = nn.InstanceNorm2d(128)
        self.batch_norm_4 = nn.InstanceNorm2d(256)

        self.fc = nn.Linear(6 * 6 * 256, 1)

        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # print("input : ", input.shape)
        x = self.conv2d_1(input)
        x = self.activation(x)
        x = self.batch_norm_1(x)

        x = self.conv2d_2(x)
        x = self.activation(x)
        x = self.batch_norm_2(x)

        x = self.conv2d_3(x)
        x = self.activation(x)
        x = self.batch_norm_3(x)

        x = self.conv2d_4(x)
        x = self.activation(x)
        x = self.batch_norm_4(x)
        # print("x : shape : ", x.shape)

        # x = self.conv2d_5(x)
        x = x.view(x.size(0), -1)
        # print("view x : ", x.shape)
        x = self.fc(x)
        x = self.sigmoid(x)

        
        return x

class Brain:
    def __init__(self, actor, critic, discriminator, json_file_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = ReplayMemory(CAPACITY, json_file_dir)
        self.td_memory = TDerrorMemory(CAPACITY)

        # actor(行動価値を求める)は２つ用意する
        self.main_actor = actor
        self.main_actor = self.init_weight(self.main_actor)
        self.target_actor = actor
        self.target_actor.load_state_dict(self.main_actor.state_dict())

        # critic(状態価値を求める)は２つ用意する
        self.main_critic = critic
        self.main_critic = self.init_weight(self.main_critic)
        self.target_critic = critic
        self.target_critic.load_state_dict(self.main_critic.state_dict())

        # Discriminator(GAN識別器)を用意
        self.discriminator = discriminator
        self.discriminator = self.init_weight(self.discriminator)

        self.actor_optimizer = optim.Adam(self.main_actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.main_critic.parameters(), lr=0.0001)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)

        self.actor_update_interval = 2

    def init_weight(self, net):
        if isinstance(net, nn.Linear):
            nn.init.kaming_uniform_(net.weight.data)
            if net.bias is not None:
                nn.init.constant_(net.bias, 0.0)
        if isinstance(net, nn.Conv2d):
            nn.init.kaming_uniform_(net.weight.data)
            if net.bias is not None:
                nn.init.constant_(net.bias, 0.0)
        return net
    
    def make_minibatch(self, episode):
        # メモリからミニバッチ分のデータを取り出す
        # if episode < 30:
        #     transitions = self.memory.sample(BATCH_SIZE)
        # else:
        #     indexes = self.td_memory.get_prioritized_indexes(BATCH_SIZE)
        #     transitions = [self.memory.memory[n] for n in indexes]
        transitions = self.memory.sample(BATCH_SIZE)

        # 各変数をメモリバッチに対する形に変形
        # transitions は　1stepごとの(state, action, state_next, reward)がBATCH_SIZE分格納されている
        # これを(state x BATCH_SIZE, action x BATCH_SIZER, state_next x BATCH_SIZE, state_next x BATCH_SIZE, reward x BATCH_SIZE)にする
        batch = Transition(*zip(*transitions))
        # print("batch : ", batch)
        # print("batch.reward : ", batch.reward)
        # print("batch.next_state : ", batch.next_state)
        # 各変数の要素をミニバッチに対応する形に変形する
        # 1x4がBATCH_SIZE分並んでいるところを　BATCH_SIZE x 4にする
        # print("batch.state : ", batch.state)
        state_batch = torch.cat(batch.state).detach().to(self.device)
        # print("state_batch : ", state_batch.size())
        action_batch = torch.cat(batch.action).detach().to(self.device)
        reward_batch = torch.cat(batch.reward).detach().to(self.device)
        next_state_batch = torch.cat(batch.next_state).detach().to(self.device)
        done_batch = batch.done

        return batch, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def actorcritic_update(self, step, episode):
        if len(self.memory) < BATCH_SIZE:
            return 0, 0
        # ミニバッチの作成
        batch, state_batch, action_batch, reward_batch, next_states, done_batch = self.make_minibatch(episode)
        #print("action_batch : ", action_batch.shape)
        # print("non_final_next_states : ", next_states.shape)
        # print("reward_batch : ", torch.sum(reward_batch))
        
        """----次の状態の価値を計算"""
        n_state_action_values = self.target_actor(next_states)
        clipped_noise = torch.clip(torch.normal(0.0, 0.2, n_state_action_values.shape), -0.5, 0.5).to(self.device)
        n_state_action_values = torch.clip(n_state_action_values + clipped_noise, -1, 1)
        # print("n_state_action_values : ", n_state_action_values.shape)

        # Actorの出力をcriticに入力して状態価値を得る
        n_q_val1, n_q_val2 = self.target_critic(next_states, n_state_action_values)
        # print("n_q_val1 : ", n_q_val1.size())
        n_q_vals = np.array([min(q1, q2) for q1, q2 in zip(n_q_val1.cpu().detach().numpy(), n_q_val2.cpu().detach().numpy())])
        # print("n_q_val : ", n_q_vals.shape)
        # print("reward_batch : ", reward_batch.size())

        # Q値の計算 (reward + gamma * n_qval)
        q_val = np.asarray([[reward] if done else [reward] + GAMMA * n_q_val for reward, done, n_q_val in zip(reward_batch, done_batch, n_q_vals)])
        q_val = torch.from_numpy(q_val.astype(np.float32)).to(self.device).detach()
        # print("q_val : ", q_val.shape)

        # print("actions : ", actions)
        """----Actorの学習----"""
        # Actorの学習は少し減らす
        if step % self.actor_update_interval == 0:
            actor_actions = self.main_actor(state_batch)
            # print("state_batch : ", state_batch.size())
            q, _ = self.main_critic(state_batch, actor_actions)
            actor_loss = -torch.mean(q)
            
            self.main_actor.train()
            self.main_critic.eval()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.main_actor.parameters(), 0.5)
            self.actor_optimizer.step()
        else:
            actor_loss = torch.tensor([0.0])

        """----Criticの学習----"""
        # Criticの学習MSEで学習
        q1, q2 = self.main_critic(state_batch, action_batch)
        loss1 = torch.mean(torch.square(q_val - q1))
        loss2 = torch.mean(torch.square(q_val - q2))
        critic_loss = loss1 + loss2

        # print("actor_loss : ", actor_loss, " | critic loss : ", critic_loss)

        self.main_actor.eval()
        self.main_critic.train()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.main_critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return float(actor_loss), float(critic_loss)
    
    def update_target_q_function(self):
        soft_tau = 0.02
        new_actor = {}
        new_critic = {}
        print("self.main_actor.state_dict() : ", self.main_actor.state_dict().keys())
        for key in self.main_actor.state_dict().keys():
            new_actor[key] = (soft_tau) * self.main_actor.state_dict()[key] + (1-soft_tau) * self.target_actor.state_dict()[key]
        for key in self.main_critic.state_dict().keys():
            new_critic[key] = (soft_tau) * self.main_critic.state_dict()[key] + (1-soft_tau) * self.target_critic.state_dict()[key]

        self.target_actor.load_state_dict(new_actor)
        self.target_critic.load_state_dict(new_critic)
    
    def discriminator_update(self, ppo_route, expert_route):
        # ---- 識別器の学習----
        real_label =  torch.ones(ppo_route.size()[0], 1, 1, 1).to(self.device)# img.size()[0]はバッチサイズのこと
        fake_label = torch.zeros(expert_route.size()[0], 1, 1, 1).to(self.device)

        # 識別器を用いて真偽を出力("生成された経路" と "用意した人間のデータ" を識別機に入力して結果を得る)
        # print("ppo_route : ", ppo_route.shape)
        fake_outputs = self.discriminator(ppo_route)
        # print("fake_outputs : ", fake_outputs.shape)
        # print("expert_route : ", expert_route.shape)
        real_outputs = self.discriminator(expert_route)

        # 識別器の入力を結合　また、 教師データとなる "偽物なら0の行列"と"本物なら1の行列"を結合
        authenticity_outputs = torch.cat((fake_outputs, real_outputs), 0)
        authenticity_targets = torch.cat((fake_label, real_label), 0)

        # バックプロパゲーション
        self.discriminator.train()
        mse_loss = nn.MSELoss()
        loss = mse_loss(authenticity_outputs, authenticity_targets) / 0.5
        self.discriminator.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

