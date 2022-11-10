import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple

import numpy as np

BATCH_SIZE = 32
CAPACITY = 10000
GAMMA = 0.99

TD_ERROR_EPSILON = 0.0001

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory():
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity # 保存しているindexを1つずらす　→　1001 % self.capacity = 1となる

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

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

class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)

        # Dueling Network
        self.fc3_adv = nn.Linear(n_mid, n_out) # Advantage側
        self.fc3_v = nn.Linear(n_mid, 1) # 価値V側

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        adv = self.fc3_adv(h2)
        val = self.fc3_v(h2).expand(-1, adv.size(1)) # valはadvと足し算するためにサイズをminibatch x 1からminibatch x 2にする
        
        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        # val + adv から　advの平均値を引き算する
        # これによって計算の偏りをなくす
        return output

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions # 制御手法を数(PurePursuit and 無限遠点PurePursuit)の2つ

        self.memory = ReplayMemory(CAPACITY)

        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)
        self.target_q_network = Net(n_in, n_mid, n_out)

        self.main_q_network = self.init_weight(self.main_q_network)
        self.target_q_network = self.init_weight(self.target_q_network)

        print(self.main_q_network)
        
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

        self.td_error_memory = TDerrorMemory(CAPACITY)

    def init_weight(self, net):
        if isinstance(net, nn.Linear):
            nn.init.kaming_uniform_(net.weight.data)
            if net.bias is not None:
                nn.init.constant_(net.bias, 0.0)
        return net

    def replay(self, episode):
        if len(self.memory) < BATCH_SIZE:
            return
        # ミニバッチの作成
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch(episode)
        # 教師信号となるQ(s_t, a_t)値を求める
        self.expected_state_action_values = self.get_expected_state_action_values()
        # 総合パラメータの更新
        self.update_main_q_network()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                # print("state : ", state)
                action = self.main_q_network(state).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]]
            )
        return action

    def make_minibatch(self, episode):
        # メモリからミニバッチ分のデータを取り出す
        if episode < 30:
            transitions = self.memory.sample(BATCH_SIZE)
        else:
            indexes = self.td_error_memory.get_prioritized_indexes(BATCH_SIZE)
            transitions = [self.memory.memory[n] for n in indexes]

        # 各変数をメモリバッチに対する形に変形
        # transitions は　1stepごとの(state, action, state_next, reward)がBATCH_SIZE分格納されている
        # これを(state x BATCH_SIZE, action x BATCH_SIZER, state_next x BATCH_SIZE, state_next x BATCH_SIZE, reward x BATCH_SIZE)にする
        batch = Transition(*zip(*transitions))

        # 各変数の要素をミニバッチに対応する形に変形する
        # 1x4がBATCH_SIZE分並んでいるところを　BATCH_SIZE x 4にする
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])


        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        self.main_q_network.eval()
        self.target_q_network.eval()

        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))

        next_state_values = torch.zeros(BATCH_SIZE)
        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        self.main_q_network.train()

        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def update_td_error_memory(self):
        self.main_q_network.eval()
        self.target_q_network.eval()

        transitions = self.memory.memory
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_action_values = self.main_q_network(state_batch).gather(1, action_batch)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        next_state_values = torch.zeros(len(self.memory))
        a_m = torch.zeros(len(self.memory)).type(torch.LongTensor)

        a_m[non_final_mask] = self.main_q_network(non_final_next_states).detach().max(1)[1]

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = self.target_q_network(non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        td_errors = (reward_batch + GAMMA * next_state_values) - state_action_values.squeeze()

        self.td_error_memory.memory = td_errors.detach().numpy().tolist()