import torch
import numpy as np

class Value:
    def __init__(self, policy):
        self.policy = policy
    def __call__(self, state, params):
        state = torch.cat((state, params), dim=-1)
        act = self.policy.actor_target(state)
        cri1, cri2 = self.policy.critic_target(state,act)
        return torch.min(cri1, cri2)
        #return min(min (cri1.numpy(), cri2.numpy())[0][0])

def get_td3_value(env_name):
    if env_name == "DartWalker2dPT-v1":
        state_dim = 25
        action_dim = 6
        max_action = 1.0
    elif env_name == "DartHopperPT-v1":
        state_dim = 16
        action_dim = 3
        max_action = 1.0

    import utils
    import policy_transfer.uposi.TD3.utils
    from policy_transfer.uposi.TD3.TD3 import TD3
    import policy_transfer.uposi.TD3.OurDDPG
    import policy_transfer.uposi.TD3.DDPG

    policy = TD3(state_dim = state_dim, action_dim = action_dim, max_action = max_action)
    policy.load("/home/hza/policy_transfer/PT/policy_transfer/uposi/TD3/models/TD3_" + env_name + "_1000")
    #policy.actor_target.to(torch.device("cpu"))
    #policy.critic_target.to(torch.device("cpu"))
    policy.actor_target.to(torch.device("cuda"))
    policy.critic_target.to(torch.device("cuda"))
    return Value(policy)


class UP:
    def __init__(self, actor_critic, ob_rms):
        self.actor_critic = actor_critic
        self.ob_rms = ob_rms
        self.device = 'cuda:0'
        self.params = None

    def reset(self):
        self.hidden = torch.zeros(
            1, self.actor_critic.recurrent_hidden_state_size, device=self.device)
        self.mask = torch.zeros(1, 1, device=self.device)

    def set_params(self, params):
        self.params = params

    def __call__(self, ob):
        assert self.params is not None
        ob = np.concatenate((ob, self.params))
        ob = torch.tensor([np.clip((ob - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + 1e-08), -10.0, 10.0)], dtype=torch.float32, device=self.device)

        _, action, _, self.hidden_state = self.actor_critic.act(ob, self.hidden, self.mask, deterministic=True)
        return action.detach().cpu().numpy()[0]


def get_up_network(env_name, num):
    import sys
    import os
    
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'PT/policy_transfer/uposi'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'PT/baselines'))
    from a2c_ppo_acktr import algo, utils
    from a2c_ppo_acktr.algo import gail
    from a2c_ppo_acktr.arguments import get_args
    from a2c_ppo_acktr.envs import make_vec_envs
    from a2c_ppo_acktr.model import Policy
    from a2c_ppo_acktr.storage import RolloutStorage
    env_name = env_name[:-5]
    if 'Dart' in env_name:
        path = f"/home/hza/policy_transfer/PT/trained_models/ppo/UP_{env_name}_{num}.pt"
    else:
        path = f"/home/hza/policy_transfer/PT/trained_models/ppo/UP_{env_name}_{num}.pt"
    result = torch.load(path, map_location=lambda a, b:torch.Storage().cuda())

    actor_critic = result[0]
    actor_critic.cuda()
    ob_rms = result[1]
    return UP(actor_critic, ob_rms)


class UP2(UP):
    def __init__(self, agent):
        self.agent = agent
        self.params = None

    def set_params(self, params):
        self.params = params

    def __call__(self, ob):
        if len(self.params.shape) == 1:
            ob = np.concatenate((ob, self.params), axis=0)[None,:]
        else:
            ob = np.concatenate((np.tile(ob,(len(self.params), 1)), self.params), axis=1)
        action = self.agent.act(ob, mode='test')
        return action.mean(axis=0)

    def reset(self):
        pass


def get_awr_network(env_name, num):
    import torch
    import sys
    sys.path.append('awr2')
    path = f'awr2/models/{env_name}'
    agent = torch.load(path)
    return UP2(agent)

def get_finetune_network(env_name, num, num_iter=21, num_proc=10):
    import torch
    import sys
    from finetune import Finetune
    sys.path.append('awr2')
    path = f'awr2/models/{env_name}'
    agent = torch.load(path)
    return Finetune(env_name, num, agent, num_iter, num_proc=num_proc)