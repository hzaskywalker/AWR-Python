import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.
"""

# define the actor network
class actor(nn.Module):
    def __init__(self, inp_dim, oup_dim, std=0.2, activation='relu', hidden_size=(128, 64)):
        super(actor, self).__init__()
        self.logstd = nn.Parameter(torch.ones(oup_dim) * np.log(std), requires_grad=False)
        self.fc1 = nn.Linear(inp_dim, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.action_out = nn.Linear(hidden_size[-1], oup_dim)
        
        self.action_out.weight.data.mul_(0.1)
        self.action_out.bias.data.mul_(0.0)

        if activation == 'relu':
            self.relu = nn.ReLU()
        elif activation == 'tanh':
            self.relu = nn.Tanh()
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        action_mean = self.action_out(x)
        return Normal(action_mean, self.logstd.exp()[None,:].expand_as(action_mean))

class critic(nn.Module):
    def __init__(self, inp_dim, activation='relu', hidden_size=(128, 64)):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(inp_dim, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.q_out = nn.Linear(hidden_size[1], 1)

        if activation == 'relu':
            self.relu = nn.ReLU()
        elif activation == 'tanh':
            self.relu = nn.Tanh()
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value[:, 0]