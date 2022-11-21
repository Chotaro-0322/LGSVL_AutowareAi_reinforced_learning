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
NUM_ADVANCED_STEP = 10
value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5
config_clip = 0.2


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class RolloutStorage:
    def __init__(self, num_steps, num_processes, obs_shape):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.observations = torch.zeros(num_steps + 1, num_processes, obs_shape[2], obs_shape[0], obs_shape[1]).to(self.device)
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(self.device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(self.device)
        self.actions = torch.zeros(num_steps, num_processes, 1).long().to(self.device)

        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
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
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]

class ActorCritic(nn.Module):
    def __init__(self, n_out):
        super(ActorCritic, self).__init__()
        self.maxpooling = nn.MaxPool2d(2)
        self.conv2d_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2d_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2d_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2d_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv2d_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv2d_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv2d_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv2d_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv2d_5_1 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.conv2d_5_2 = nn.Conv2d(1024, 512, 3, stride=1, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2)

        self.convtrans2d_6_1 = nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=1)
        self.convtrans2d_6_2 = nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1)
        self.convtrans2d_7_1 = nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1)
        self.convtrans2d_7_2 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1)
        self.convtrans2d_8_1 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1)
        self.convtrans2d_8_2 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1)
        self.convtrans2d_9_1 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1)
        self.convtrans2d_9_2 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.convtrans2d_9_3 = nn.ConvTranspose2d(64, 3, 1, stride=1)

        self.batch_norm_s32 = nn.InstanceNorm2d(32)
        self.batch_norm_s64 = nn.InstanceNorm2d(64)
        self.batch_norm_s128 = nn.InstanceNorm2d(128)
        self.batch_norm_s256 = nn.InstanceNorm2d(256)
        self.batch_norm_s512 = nn.InstanceNorm2d(512)
        self.batch_norm_s1024 = nn.InstanceNorm2d(1024)
        self.dropout = nn.Dropout2d(0.5)

        self.activation = nn.LeakyReLU(0.01)

        self.fc = nn.Linear(1024 * 5 * 5, 512)
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, n_out)

        self.activation = nn.LeakyReLU(0.01)
        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpooling = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv2d_1_1(x)
        x = self.activation(x)
        x = self.batch_norm_s64(x)
        x = self.conv2d_1_2(x)
        x = self.activation(x)
        x = self.batch_norm_s64(x)
        x_d0 = x
        x = self.maxpooling(x)

        x = self.conv2d_2_1(x)
        x = self.activation(x)
        x = self.batch_norm_s128(x)
        x = self.conv2d_2_2(x)
        x = self.activation(x)
        x = self.batch_norm_s128(x)
        x_d1 = x
        x = self.maxpooling(x)

        x = self.conv2d_3_1(x)
        x = self.activation(x)
        x = self.batch_norm_s256(x)
        x = self.conv2d_3_2(x)
        x = self.activation(x)
        x = self.batch_norm_s256(x)
        x_d2 = x
        x = self.maxpooling(x)
        
        x = self.conv2d_4_1(x)
        x = self.activation(x)
        x = self.batch_norm_s512(x)
        x = self.conv2d_4_2(x)
        x = self.activation(x)
        x = self.batch_norm_s512(x)
        x_d3 = x
        x = self.maxpooling(x)
        
        x = self.conv2d_5_1(x)
        x = self.activation(x)
        x = self.batch_norm_s1024(x)
        x = self.conv2d_5_2(x)
        x = self.activation(x)
        x = self.batch_norm_s1024(x)

        middle_output = x
        
        x = self.upsample(x)
        x = torch.cat([x_d3, x], 1)
        
        x = self.convtrans2d_6_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.batch_norm_s512(x)
        x = self.convtrans2d_6_2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.batch_norm_s256(x)

        x = self.upsample(x)
        x = torch.cat([x_d2, x], 1)
        
        x = self.convtrans2d_7_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.batch_norm_s256(x)
        x = self.convtrans2d_7_2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.batch_norm_s128(x)

        x = self.upsample(x)
        x = torch.cat([x_d1, x], 1)

        x = self.convtrans2d_8_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.convtrans2d_8_2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.upsample(x)
        x = torch.cat([x_d0, x], 1)
        
        x = self.convtrans2d_9_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.convtrans2d_9_2(x)
        x = self.activation(x)
        x = self.dropout(x)
        actor_output = self.convtrans2d_9_3(x)

        middle_output = middle_output.view(middle_output.size(0), -1)
        middle_output = self.fc(middle_output)
        critic_output = self.critic(middle_output)

        return critic_output, actor_output

    def act(self, x):
        print("x size : ", x.size())
        value, actor_output = self(x)
        action_probs = F.softmax(actor_output, dim=1)
        # print("action_probs : ", action_probs)
        action = action_probs.multinomial(num_samples=1)
        # print("action : ", action.squeeze())
        max_action_probs = action_probs.max(1)[0] # 選ばれた行動の確率を求める
        # print("max_action_probs : ", max_action_probs)
        return action, max_action_probs

    def get_value(self, x):
        value, actor_output = self(x)
        return value

    def evaluate_actions(self, x, actions):
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)

        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)
        # print("action : ", action.squeeze())
        # print("action_prob : ", action_probs)
        # print("max_action_probs : ", action_probs.max(1)[0])
        max_action_probs = action_probs.max(1)[0] # 選ばれた行動の確率を求める

        action_log_probs = log_probs.gather(1, actions)

        probs = F.softmax(actor_output, dim=1)
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy, max_action_probs

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 64, 2, stride=2)
        self.conv2d_2 = nn.Conv2d(64, 128, 2, stride=2)
        self.conv2d_3 = nn.Conv2d(128, 256, 2, stride=2)
        self.conv2d_4 = nn.Conv2d(256, 512, 2, stride=2)
        self.conv2d_5 = nn.Conv2d(512, 1, 1, stride=1) # ボルトネック層

        self.batch_norm_1 = nn.InstanceNorm2d(64)
        self.batch_norm_2 = nn.InstanceNorm2d(128)
        self.batch_norm_3 = nn.InstanceNorm2d(256)
        self.batch_norm_4 = nn.InstanceNorm2d(512)

        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
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

        x = self.conv2d_5(x)
        
        return x

class Brain:
    def __init__(self, actor_critic, discriminator):
        self.actor_critic = actor_critic
        self.actor_critic = self.init_weight(self.actor_critic)

        self.discriminator = discriminator
        self.discriminator = self.init_weight(self.discriminator)

        self.actorcritic_optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.01)

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
        obs_shape = rollouts.observations.size()[2]
        # print("obs_shape : ", obs_shape)
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy, action_probs = self.actor_critic.evaluate_actions(rollouts.observations[:-1].view(-1, obs_shape), rollouts.actions.view(-1, 1))

        values = values.view(num_steps, num_processes, 1) # torch.Size([5, 1, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        action_probs = action_probs.view(num_steps, num_processes, 1)
        # print("action_log_probs : ", action_probs.size())
        if first_episode == True:
            ratio = torch.ones(num_steps, num_processes, 1)
            pass
        else:
            print("action_probs : ", action_probs.squeeze())
            print("old_actin_probs : ", old_action_probs.squeeze())
            ratio = torch.div(action_probs, old_action_probs.detach() + 0.00001)
            ratio = action_probs / old_action_probs
        # print("action_probs : ", action_probs)
        # print("old_action_probs : ", old_action_probs)
        # print("ratio : ", ratio)

        advantages = rollouts.returns[:-1] - values # torch.size([5, 1, 1])

        value_loss = advantages.pow(2).mean()

        clipped_ratio = torch.clip(ratio, (1 - config_clip), (1 + config_clip))
        clipped_loss = torch.min(ratio*advantages.detach(), clipped_ratio*advantages.detach()).mean()
        print("ratio : ", ratio.squeeze(), "\n----------------------------------------\n")
        # print("clipped_ratio : ", clipped_ratio)
        # action_gain = (action_log_probs*advantages.detach()).mean() # これはa2cのloss
        # print("action_gain : ", action_gain)

        # print("value_loss : ", value_loss_coef * value_loss)
        # print("clipped_loss : ", clipped_loss)
        # print("entropy : ", entropy_coef * entropy)

        total_loss = (value_loss * value_loss_coef - clipped_loss - entropy * entropy_coef)

        self.actor_critic.train()
        self.actorcritic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)

        self.actorcritic_optimizer.step()
    
    def discriminator_update(self, ppo_route, expert_route):
        # ---- 識別器の学習----
        real_label =  torch.ones(ppo_route.size()[0], 1, 16, 16).to(self.device)# img.size()[0]はバッチサイズのこと
        fake_label = torch.zeros(expert_route.size()[0], 1, 16, 16).to(self.device)

        # 識別器を用いて真偽を出力("生成された経路" と "用意した人間のデータ" を識別機に入力して結果を得る)
        fake_outputs = self.discriminator(ppo_route)
        real_outputs = self.discriminator(expert_route)

        # 識別器の入力を結合　また、 教師データとなる "偽物なら0の行列"と"本物なら1の行列"を結合
        authenticity_outputs = torch.cat((fake_outputs, real_outputs), 0)
        authenticity_targets = torch.cat((fake_label, real_label), 0)

        # バックプロパゲーション
        mse_loss = nn.MSELoss()
        loss = mse_loss(authenticity_outputs, authenticity_targets) / 0.5
        self.discriminator.zero_grad()
        loss.backward()
        self.discriminator.step()

