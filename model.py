import numpy as np
import torch

from envs import make
from utils import DataParallel


def get_state(env):
    return np.append(env.state_vector(), [env._elapsed_steps])

def get_params(env):
    return env.env.param_manager.get_simulator_parameters()

def set_params(env, param):
    env.env.param_manager.set_simulator_parameters(param)

def set_state(env, state, param=None):
    if param is not None:
        if param.size > 0:
            env.env.param_manager.set_simulator_parameters(param)
    env.env.cur_step = env._elapsed_steps = int(state[-1])
    state = state[:-1]
    env.set_state_vector(state)


class Rollout:
    # rollout return the observation but not the states..
    def __init__(self, make, env_name, num, stochastic_obs, done=True):
        self.env = make(env_name, num)
        self.env.reset()

        # TODO:close noise
        self.env.env.noisy_input = stochastic_obs
        self.done = done

    def __call__(self, params, s, a):
        rewards = []

        obs, mask = None, None
        batch_size = len(s)

        batch_idx = 0
        for p, s, a in zip(params, s, a):
            self.env.reset() # perhaps we need to clear the contact results for dart...
            set_state(self.env, s, p)
            reward = 0

            for idx, action in enumerate(a):
                if action.max() < -10000:
                    break
                s, r, done, _ = self.env.step(action)
                reward += r
                if obs is None:
                    obs = np.zeros((batch_size, len(a), len(s)))
                    mask = np.zeros((batch_size, len(a)), dtype=np.int32)

                obs[batch_idx, idx] = s
                mask[batch_idx, idx] = 1

                if done and self.done:
                    break
            rewards.append(reward)
            batch_idx += 1
        return np.array(rewards), np.array(obs), np.array(mask)


def make_parallel(num_proc, env_name, num, stochastic=False, done=True):
    return DataParallel(num_proc, Rollout, make, env_name, num, stochastic, done=done)