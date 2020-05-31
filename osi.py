# different osi model
# osi model have the following form:
#   init(trajectories)
#   update()
import numpy as np
from optim import CEM
import torch
from collections import deque

class CEMOSI:
    def __init__(self, model, init_mean, iter_num, num_mutation, num_elite,
        *args, std=0.3, queue_size=5, **kwargs):
        self.model = model
        self.cem = CEM(self.cost, iter_num, num_mutation, num_elite, *args, std=std, **kwargs)

        # pass
        # each trajectory is made of the init state, action sequence and obs sequence and the corresponding masks..
        self.states = [] #(state, a, obs, mask)
        self.actions = []
        self.obs = []
        self.masks = []
        self._params = init_mean
        self.init = init_mean

    def cost(self, scene, params):
        #return self.optim()
        n_param = len(params)
        n_traj = len(self.states)
        device = params.device

        params = params.detach().cpu().numpy().clip(0, np.inf)
        negative_mask = (params.max(axis=1) < 0)

        params = np.tile(params[:, None], (1, n_traj, 1))
        states = np.tile(np.array([self.states]), (n_param, 1, 1))
        actions = np.tile(np.array([self.actions]), (n_param, 1, 1))

        params = params.reshape(-1, params.shape[-1])
        states = states.reshape(n_param * n_traj, states.shape[-1])
        actions = actions.reshape(n_param * n_traj, -1, actions.shape[-1])

        reward, obs, mask = self.model(params, states, actions)
        obs = obs.reshape(n_param, n_traj, *obs.shape[-2:])
        error = np.linalg.norm((obs - np.array(self.obs)[None,:])**2, axis=-1).sum(axis=(1, 2)) 

        if negative_mask.any():
            error[negative_mask] = 1e9
        return torch.tensor(error, dtype=torch.float, device=device)

    def update(self, state, obs, action, mask):
        self.states.append(state)
        self.actions.append(action)
        self.obs.append(obs)
        self.masks.append(mask)
    
    def reset(self):
        self.states = []
        self.actions = []
        self.obs = []
        self.masks = []
        self._params = self.init

    def optim(self):
        mean = torch.tensor(self._params, dtype=torch.float, device='cuda:0')
        self._params = self.cem(None, mean).detach().cpu().numpy()

    def get_params(self):
        # optim
        self.optim()
        return self._params


def test_up_osi():
    from evaluation import osi_eval
    from networks import get_up_network
    from model import make_parallel, make, get_params

    env_name = 'DartHopperPT-v1'
    num = 5

    policy_net = get_up_network(env_name, num)

    model = make_parallel(10, env_name, num=num, stochastic=False)
    env = make(env_name, num=num, resample_MP=True, stochastic=False)

    params = get_params(env)

    mean_params = policy_net.ob_rms.mean[-len(params):]
    mean_params = np.array([0.5] * len(params))
    osi = CEMOSI(model, mean_params,
        iter_num=20, num_mutation=100, num_elite=10, std=0.3)
    policy_net.set_params(mean_params)

    osi_eval(env, osi, policy_net, num_init_traj=1, max_horizon=20, eval_episodes=10, use_state=False, print_timestep=10000)


if __name__ == '__main__':
    test_up_osi()