# run the ablation study probelm
import numpy as np
from optim import CEM
import torch
from collections import deque
from evaluation import osi_eval, online_osi
from networks import get_up_network, get_awr_network, get_finetune_network
from model import make_parallel, make, get_params, set_params
import random
from osi import CEMOSI


def mytest(env_name, eval_episode=10, num_init_traj=1, max_horizon=15, ensemble=1, gt=0, finetune=False, finetune_iter=41, finetune_proc=10, cem_iter=20):
    NUMS = {
        'HalfCheetahPT-v2': 6,
        'HopperPT-v2': 5,
        'Walker2dPT-v2': 8,
    }

    num = NUMS[env_name]

    if not finetune:
        policy_net = get_awr_network(env_name, num)
    else:
        policy_net = get_finetune_network(env_name, num, num_iter=finetune_iter, num_proc=finetune_proc)

    model = make_parallel(10, env_name, num=num, stochastic=False)
    env = make(env_name, num=num, resample_MP=True, stochastic=False)

    params = get_params(env)
    mean_params = np.array([0.5] * len(params))
    osi = CEMOSI(model, mean_params,
        iter_num=cem_iter, num_mutation=100, num_elite=10, std=0.3)

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


def exp(env_name):
    #env_name = 'HopperPT-v2'
    #env_name = 'HalfCheetahPT-v2'
    #env_name = 'Walker2dPT-v2'
    results = {}
    eval_episode = 10
    #results['gt']=mytest(env_name, gt=1, eval_episode=eval_episode, num_init_traj=1, finetune=0)
    #results['gt2']=mytest(env_name, gt=1, eval_episode=eval_episode, num_init_traj=1, finetune=1, finetune_iter=11)
    #results['gt2']=mytest(env_name, gt=1, eval_episode=eval_episode, num_init_traj=1, finetune=1, finetune_iter=0, finetune_proc=1)
    #results['more_traj']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=1)
    results['ensemble_finetue']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=1, ensemble=5, finetune=1, finetune_iter=41, finetune_proc=10)
    results['gt']=mytest(env_name, gt=1, eval_episode=eval_episode, num_init_traj=1, ensemble=1)
    results['osi']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=1, ensemble=1)
    results['ensemble']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=1, ensemble=5)

    import json
    with open(f'{env_name}.json', 'w') as f:
        json.dump(results, f)

    
def ablation(env_name):
    results = {}

    eval_episode = 30
    results['traj_1']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=1, ensemble=1)
    results['traj_5']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=1)
    results['traj_10']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=10, ensemble=1)
    results['traj_20']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=20, ensemble=1)

    results['cem_1']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=1, cem_iter=1)
    results['cem_5']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=1, cem_iter=5)
    results['cem_10']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=1, cem_iter=10)
    results['cem_20']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=1, cem_iter=20)
    results['cem_25']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=1, cem_iter=25)
    results['cem_30']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=1, cem_iter=30)

    results['ensemble_1']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=1)
    results['ensemble_5']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=5)
    results['ensemle_10']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=10)
    results['ensemble_15']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=15)
    results['ensemble_20']=mytest(env_name, gt=0, eval_episode=eval_episode, num_init_traj=5, ensemble=25)

    import json
    with open(f'{env_name}_ablation.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    from multiprocessing import set_start_method
    set_start_method('spawn')
    import sys
    exp(sys.argv[1])
    #ablation(sys.argv[1])