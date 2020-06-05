import gym
import sys
import numpy as np
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PT'))
import policy_transfer.envs


def make(env_name, num, seed = 1000, stochastic=True, resample_MP=False, train_UP=False):
    env = None
    if env_name == "DartHopperPT-v1" and num == 2:
        env = make_env(env_name, [0,5], seed)
    elif env_name == "DartHopperPT-v1" and num == 5:
        env = make_env(env_name, [0,1,2,5,9], seed)
    elif env_name == "HopperPT-v2" and num == 5:
        env = make_env(env_name, [0,1,2,5,8], seed)
    elif env_name == "HopperPT-v3" and num == 5:
        env = make_env(env_name, [0,1,2,5,8], seed)
    elif env_name == "DartHopperPT-v1" and num == 10:
        env = make_env(env_name, [0,1,2,3,4,5, 6,7,8, 9], seed)
    elif env_name == "DartWalker2dPT-v1" and num == 8:
        env = make_env(env_name, [7,8,9,10,11,12, 13,14], seed)
    elif env_name == "Walker2dPT-v2" and num == 8:
        env = make_env(env_name, [4, 5, 6, 7, 8, 9, 10, 11], seed)
    elif env_name == "Walker2dPT-v4" and num == 3:
        env = make_env("Walker2dPT-v2", [4, 5, 6], seed)
    elif env_name == "Walker2dPT-v3" and num == 8:
        env = make_env(env_name, [4, 5, 6, 7, 8, 9, 10, 11], seed)
    elif env_name == "DartWalker2dPT-v1" and num == 15:
        env = make_env(env_name, [0,1,2,3,4,5,6,7,8,9,10,11,12, 13,14], seed)
    elif env_name == "DartHalfCheetahPT-v1" and num == 8:
        env = make_env(env_name, [0,1,2,3,4,5,6,7], seed)
    elif env_name == "HalfCheetahPT-v2" and num == 7:
        env = make_env(env_name, [0,1,2,3,4,5,6], seed)
    else:
        raise NotImplementedError(f"{env_name} and num={num} is not implemented")
    env.env.noisy_input = stochastic
    env.env.resample_MP = resample_MP
    env.env.train_UP = train_UP
    if train_UP:
        import gym
        inp_dim = env.env.observation_space.shape[0]+num
        env.observation_space = env.env.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(inp_dim,), dtype=np.float32)

    return env
    

def make_env(env_name, dyn_params, seed=1000):
    env = gym.make(env_name)
    env.seed(seed)
    env.env.param_manager.activated_param = dyn_params
    env.env.param_manager.controllable_param = dyn_params
    env.reset()
    return env

if __name__ == '__main__':
    env = make("DartWalker2dPT-v1", 15)
    print(env.env.param_manager.get_simulator_parameters())
    env = make("DartHopperPT-v1", 10)
    print(env.env.param_manager.get_simulator_parameters())