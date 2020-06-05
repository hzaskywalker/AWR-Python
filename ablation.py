# run the ablation study probelm
import numpy as np
from optim import CEM
import torch
from collections import deque
from evaluation import osi_eval, online_osi
from networks import get_up_network, get_awr_network
from model import make_parallel, make, get_params, set_params
import random
from osi import CEMOSI


def mytest(env_name, eval_episode=10, num_init_traj=1, max_horizon=15, ensemble=1, gt=0):
    NUMS = {
        'HalfCheetahPT-v2': 7,
        'HopperPT-v2': 5,
        'Walker2dPT-v2': 8,
    }

    num = NUMS[env_name]

    policy_net = get_awr_network(env_name, num)

    model = make_parallel(10, env_name, num=num, stochastic=False)
    env = make(env_name, num=num, resample_MP=True, stochastic=False)

    params = get_params(env)
    mean_params = np.array([0.5] * len(params))
    osi = CEMOSI(model, mean_params,
        iter_num=20, num_mutation=100, num_elite=10, std=0.3)
    policy_net.set_params(mean_params)


    rewards, dist = online_osi(env, osi, policy_net, num_init_traj=num_init_traj, max_horizon=max_horizon, eval_episodes=eval_episode, use_state=False, print_timestep=10000, resample_MP=True, ensemble=ensemble, online=0, gt=gt)
    rewards = np.array(rewards)
    print('l2 distance', dist)
    print('rewards', rewards)
    return {
        'mean': rewards.mean(),
        'std': rewards.std(),
        'min': rewards.min(),
        'max': rewards.max(),
        'dist': dist.mean(),
    }


def exp():
    #env_name = 'HopperPT-v2'
    env_name = 'HalfCheetahPT-v2'
    results = {}
    eval_episode = 5
    #results['gt']=mytest(env_name, gt=1, eval_episode=eval_episode, num_init_traj=1)
    results['more_traj']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=1)
    #results['osi']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=1, ensemble=1)
    #results['ensemble']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=1, ensemble=5)

    import json
    with open(f'{env_name}.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    exp()