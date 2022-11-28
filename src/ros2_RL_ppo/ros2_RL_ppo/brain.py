import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple
from torch.distributions import Normal

import numpy as np

BATCH_SIZE = 32
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

def compute_pi(mean, stddev, actions):
    stddev = stddev ** 2
    a1 = (2 * np.pi * stddev) ** (-1/2)
    a2 = torch.exp(-((actions - mean) ** 2) / (2 * stddev))
    return a1 * a2

def compute_logpi(mean, stddev, action):
    a1 = -0.5 * np.log(2*np.pi)
    a2 = -torch.log(stddev)
    a3 = -0.5 * (((action - mean) / stddev) ** 2)
    return a1 + a2 + a3

class RolloutStorage:
    def __init__(self, num_steps, num_processes, obs_shape):
        self.observations = torch.zeros(num_steps + 1, num_processes, obs_shape)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()

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

class Actor(nn.Module):
    def __init__(self, n_in, n_mid, n_out, action_space_high, action_space_low):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)

        # Actor
        self.pi_mean = nn.Linear(n_mid, n_out) # Advantage側
        self.stddev = nn.Linear(n_mid, n_out)

        self.action_centor = (action_space_high + action_space_low)/2
        self.action_scale = action_space_high - self.action_centor

    def forward(self, x):
        # print("x: shape : ", x.size())
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        mean_output = self.pi_mean(h2)
        
        stddev_output = self.stddev(h2)
        stddev_output = torch.clamp(stddev_output, min=-20, max=2)
        
        return mean_output, stddev_output

    def act(self, x):
        mean, stddev = self(x)
        mean = mean.squeeze()
        stddev = stddev.squeeze()
        stddev = stddev.exp() # min=-20, max=2 のとき 2.06e-9〜7
        action_org = torch.normal(mean=mean, std=stddev, size=mean.size()) # meanの数字からdtddev分(2.06e-9〜7)程度に変動させる
        # print("action_org : ", action_org)
        action_org = torch.tanh(action_org)
        env_action = action_org * self.action_scale + self.action_centor
        action_probs = torch.tanh(mean) * self.action_scale + self.action_centor
        # print("アクションの確率 : ", action_probs)
        
        return env_action, action_probs

    def evaluate_actions(self, x, actions):
        mean, _ = self(x)
        print("mean : \n", mean)
        action_probs = torch.tanh(mean) * self.action_scale + self.action_centor
        logpi = torch.log(action_probs)
        pi = action_probs
        entropy = -(logpi * pi).sum(-1).mean()

        return entropy, pi

class Critic(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)

        # Critic
        self.critic = nn.Linear(n_mid, 1) # 価値V側

    def forward(self, x):
        # print("x: shape : ", x.size())
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        critic_output = self.critic(h2)
        
        return critic_output

    def get_value(self, x):
        value = self(x)
        return value

class Brain:
    def __init__(self, actor, critic):
        self.actor = actor
        self.actor = self.init_weight(self.actor)

        self.critic = critic
        self.critic = self.init_weight(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.01)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.01)

    def init_weight(self, net):
        if isinstance(net, nn.Linear):
            nn.init.kaming_uniform_(net.weight.data)
            if net.bias is not None:
                nn.init.constant_(net.bias, 0.0)
        return net
    
    def update(self, rollouts, old_action_probs, first_episode=False): # first_episode==Trueのとき、それは最初の試行であり、self.old_r_thera = torch.tensor([1, 1, 1, 1, 1...])を使う
        obs_shape = rollouts.observations.size()[2]
        # print("obs_shape : ", obs_shape)
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        entropy, action_probs = self.actor.evaluate_actions(rollouts.observations[:-1].view(-1, obs_shape), rollouts.actions.view(-1, 1))
        values = self.critic.get_value(rollouts.observations[:-1].view(-1, obs_shape))

        values = values.view(num_steps, num_processes, 1) # torch.Size([5, 1, 1])
        action_probs = action_probs.view(num_steps, num_processes, 1)
        if first_episode == True:
            ratio = torch.ones(num_steps, num_processes, 1)
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

        print("clipped_loss : ", clipped_loss)
        print("value_loss : ", value_loss * value_loss_coef)
        print("entropy_loss : ", entropy * entropy_coef)

        total_loss = (value_loss * value_loss_coef - clipped_loss - entropy * entropy_coef)

        self.actor.train()
        self.critic.train()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()