import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple

import numpy as np

BATCH_SIZE = 32
NUM_STACK_FRAME = 4
CAPACITY = 10000
GAMMA = 0.99

TD_ERROR_EPSILON = 0.0001

NUM_PROCESSES = 1
NUM_ADVANCED_STEP = 50
value_loss_coef = 0.5
entropy_coef = 0.5
max_grad_norm = 0.5
config_clip = 0.2


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class RolloutStorage:
    def __init__(self, num_steps, num_processes, obs_shape_1,  obs_shape_2, obs_shape_3, action_num):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.observations = torch.zeros(num_steps + 1, num_processes, obs_shape_1, obs_shape_2, obs_shape_3).to(self.device)
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(self.device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(self.device)
        self.actions = torch.zeros(num_steps, num_processes, action_num).long().to(self.device)

        self.returns = torch.zeros(num_steps + 1, num_processes, 1).to(self.device)
        self.index = 0
    
    def insert(self, current_obs, action, reward, mask):
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        self.returns[-1] = next_value
        # print("next_value : ", next_value)
        # print("self.returns : ", self.returns)
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]

class Actor(nn.Module):
    def __init__(self, n_in, n_mid, n_out, action_space_high, action_space_low):
        super(Actor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc = nn.Linear(64 * 9 * 9, 512)

        # Actor
        self.pi_mean = nn.Linear(512, n_out) # Advantage側
        self.stddev = nn.Linear(512, n_out)

        # print("action_space_high : ", action_space_high)
        # print("action_space_low : ", action_space_low)
        self.action_center = (action_space_high + action_space_low)/2
        self.action_scale = action_space_high - self.action_center
        self.action_range = action_space_high - action_space_low
        # print("self.action_scale : ", self.action_scale)
        # print("self.action_center : ", self.action_center)
        # print("self.action_range : ", self.action_range)

    def forward(self, x):
        # print("x: shape : ", x.size())
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))
        # print("x size : ", x.size())
        x = x.view(x.size(0), -1)
        # print("x size flatten : ", x.size())
        x = F.relu(self.fc(x))
        mean_output = self.pi_mean(x)
        
        stddev_output = self.stddev(x)
        stddev_output = torch.clamp(stddev_output, min=-20, max=2)
        
        return mean_output, stddev_output

    def act(self, x):
        mean, stddev = self(x)
        mean = mean.squeeze()
        stddev = stddev.squeeze()
        stddev = stddev.exp() # min=-20, max=2 のとき 2.06e-9〜7
        # print("mean.size : ", mean.size())
        # print("stddev.size : ", stddev.size())
        action_org = torch.normal(mean=mean, std=stddev) # meanの数字からdtddev分(2.06e-9〜7)程度に変動させる
        # print("action_org : ", action_org.shape)
        action_org = torch.tanh(action_org)
        env_action = action_org * self.action_scale + self.action_center
        # print("mean device : \n", mean.device)
        # action_probs = (torch.tanh(mean) * self.action_scale + self.action_center) / self.action_range
        action_probs = torch.sigmoid(mean) + 0.000000001
        # print("アクションの確率 : ", action_probs)
        
        return env_action, action_probs

    def evaluate_actions(self, x, actions):
        # print("x : ", x.shape)
        mean, _ = self(x)
        print("mean : ", mean)
        # action_probs = (torch.tanh(mean) * self.action_scale + self.action_center) / self.action_range
        action_probs = torch.sigmoid(mean) + 0.000000001
        # action_probs = torch.tanh(mean)
        # print("action probs : ", action_probs)
        logpi = torch.log(action_probs)
        pi = action_probs
        # print("logpi : ", logpi)
        entropy = -(logpi * pi).sum(-1).mean()

        return entropy, pi

class Critic(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Critic, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc = nn.Linear(64 * 9 * 9, 512)

        # Critic
        self.critic = nn.Linear(512, 1) # 価値V側
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("x: shape : ", x.size())
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        critic_output = self.critic(x)
        critic_output = torch.clamp(critic_output, min=0, max=20)
        
        return critic_output

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
        # x = self.batch_norm_1(x)

        x = self.conv2d_2(x)
        x = self.activation(x)
        # x = self.batch_norm_2(x)

        x = self.conv2d_3(x)
        x = self.activation(x)
        # x = self.batch_norm_3(x)

        x = self.conv2d_4(x)
        x = self.activation(x)
        # x = self.batch_norm_4(x)
        # print("x : shape : ", x.shape)

        # x = self.conv2d_5(x)
        x = x.view(x.size(0), -1)
        # print("view x : ", x.shape)
        x = self.fc(x)
        x = self.sigmoid(x)

        
        return x

class Brain:
    def __init__(self, actor, critic, discriminator):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = actor
        self.actor = self.init_weight(self.actor)

        self.critic = critic
        self.critic = self.init_weight(self.critic)

        self.discriminator = discriminator
        self.discriminator = self.init_weight(self.discriminator)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0001)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)

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
    
    def actorcritic_update(self, rollouts, old_action_probs, first_episode=False): # first_episode==Trueのとき、それは最初の試行であり、self.old_r_thera = torch.tensor([1, 1, 1, 1, 1...])を使う
        obs_shape = rollouts.observations.size()
        action_num = rollouts.actions.size()[2]
        # print("obs_shape : ", obs_shape)
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES
        # print("rollouts.observations[:-1] : ", rollouts.observations[:-1].shape)
        entropy, action_probs = self.actor.evaluate_actions(rollouts.observations[:-1].view(-1, obs_shape[1], obs_shape[3], obs_shape[4]), rollouts.actions.view(-1, 1))
        values = self.critic.get_value(rollouts.observations[:-1].view(-1, obs_shape[1], obs_shape[3], obs_shape[4]))

        values = values.view(num_steps, num_processes, 1) # torch.Size([5, 1, 1])
        # action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        action_probs = action_probs.view(num_steps, num_processes, action_num)
        # print("action_log_probs : ", action_probs.size())
        if first_episode == True:
            ratio = torch.ones(num_steps, num_processes, action_num).to(self.device)
            pass
        else:
            # print("action_probs : ", action_probs.squeeze())
            # print("old_actin_probs : ", old_action_probs.squeeze())
            ratio = torch.div(action_probs, old_action_probs.detach() + 0.00001)
            ratio = action_probs / old_action_probs

        advantages = rollouts.returns[:-1] - values # torch.size([5, 1, 1])

        value_loss = advantages.pow(2).mean()

        clipped_ratio = torch.clip(ratio, (1 - config_clip), (1 + config_clip))
        clipped_loss = torch.min(ratio*advantages.detach(), clipped_ratio*advantages.detach()).mean()

        print("Loss List : [value_loss : {}, clipped_loss : {}, entropy_loss : {}] ".format(value_loss * value_loss_coef, clipped_loss, entropy * entropy_coef))

        # total_loss = (value_loss * value_loss_coef - clipped_loss - entropy * entropy_coef)
        total_loss = value_loss * value_loss_coef - clipped_loss - entropy * entropy_coef

        self.actor.train()
        self.critic.train()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
    
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

