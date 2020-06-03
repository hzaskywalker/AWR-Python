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
    def __init__(self, inp_dim, oup_dim, action_space, std=0.2):
        super(actor, self).__init__()
        self.min = nn.Parameter(torch.tensor(action_space.low), requires_grad=False)
        self.max = nn.Parameter(torch.tensor(action_space.high), requires_grad=False)
        self.logstd = nn.Parameter(torch.ones(self.min.shape) * np.log(std), requires_grad=False)

        self.fc1 = nn.Linear(inp_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # TODO: action network initialization
        self.action_out = nn.Linear(64, oup_dim)
        
        self.action_out.weight.data.mul_(0.1)
        self.action_out.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #actions = self.max_action * torch.tanh(self.action_out(x))
        #return actions
        action_mean = self.action_out(x) * (self.max - self.min)[None, :] + self.min[None, :]
        return Normal(action_mean, self.logstd.exp()[None,:].expand_as(action_mean))

class critic(nn.Module):
    def __init__(self, inp_dim):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value