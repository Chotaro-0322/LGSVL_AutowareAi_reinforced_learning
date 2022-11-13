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

NUM_PROCESSES = 1
NUM_ADVANCED_STEP = 10
value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5
config_clip = 0.2


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

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

class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)

        # Actor and Critic
        self.actor = nn.Linear(n_mid, n_out) # Advantage側
        self.critic = nn.Linear(n_mid, 1) # 価値V側

    def forward(self, x):
        # print("x: shape : ", x.size())
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        critic_output = self.critic(h2)
        actor_output = self.actor(h2)
        
        return critic_output, actor_output

    def act(self, x):
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

class Brain:
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic
        self.actor_critic = self.init_weight(self.actor_critic)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)

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
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)

        self.optimizer.step()