# different osi model
# osi model have the following form:
#   init(trajectories)
#   update()
import numpy as np
from optim import CEM
import torch
from collections import deque
from evaluation import osi_eval
from networks import get_up_network
from model import make_parallel, make, get_params, set_params


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

        params = params.detach().cpu().numpy()
        negative_mask = (params.max(axis=-1) < -0.05)
        params = params.clip(-0.05, np.inf)

        params = np.tile(params[:, None], (1, n_traj, 1))
        states = np.tile(np.array([self.states]), (n_param, 1, 1))
        actions = np.tile(np.array([self.actions]), (n_param, 1, 1))

        params = params.reshape(-1, params.shape[-1])
        states = states.reshape(n_param * n_traj, states.shape[-1])
        actions = actions.reshape(n_param * n_traj, -1, actions.shape[-1])

        reward, obs, mask = self.model(params, states, actions)
        obs = obs.reshape(n_param, n_traj, *obs.shape[-2:])

        norm = obs - np.array(self.obs)[None,:]
        norm = norm / np.abs(np.array(self.obs))[None, :].clip(1, np.inf)
        norm = np.linalg.norm(norm, axis=-1)
        error = norm.sum(axis=(1, 2)) 

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
        self._params = self.cem(None, mean).detach().cpu().numpy().clip(-0.05, np.inf)
        self._loss = float(self.cem.loss)

    def get_params(self):
        # optim
        self.optim()
        return self._params

    def find_min(self, ensemble_num):
        params = None
        loss = 0

        pp = 0
        for i in range(ensemble_num):
            self._params= self.init
            out = self.get_params()
            if self._loss < loss or params is None:
                loss = self._loss
                params = out
            pp = pp + params
        return params
        #return pp/ensemble_num


class DiffOSI(CEMOSI):
    def __init__(self, model, iters):
        raise NotImplementedError

    def optim(self):
        mean = self._params
        for i in range(10):
            to_calc = [self._params]
            mean = torch.tensor(to_calc, dtype=torch.float, device='cuda:0')


def test_up_osi():
    env_name = 'DartHopperPT-v1'
    num = 5

    policy_net = get_up_network(env_name, num)

    model = make_parallel(30, env_name, num=num, stochastic=False)
    env = make(env_name, num=num, resample_MP=True, stochastic=False)

    params = get_params(env)
    #set_params(env, [0.55111654,0.55281674,0.46355396,0.84531834,0.58944066])
    set_params(env, [0.31851129, 0.93941556, 0.02147825, 0.43523052, 1.02611646])
    set_params(env, [0.94107358, 0.77519005, 0.44055224, 0.9369426, -0.03846457])
    set_params(env, [0.05039606, 0.14680257, 0.56502066, 0.25723492, 0.73810709])

    mean_params = policy_net.ob_rms.mean[-len(params):]
    mean_params = np.array([0.5] * len(params))
    osi = CEMOSI(model, mean_params,
        iter_num=20, num_mutation=100, num_elite=10, std=0.3)
    policy_net.set_params(mean_params)


    osi_eval(env, osi, policy_net, num_init_traj=1, max_horizon=20, eval_episodes=50, use_state=False, print_timestep=10000, resample_MP=True)


def test_cem_osi():
    env_name = 'DartHopperPT-v1'
    num = 5

    from networks import get_td3_value
    value_net = get_td3_value(env_name)

    from policy import POLO, add_parser
    import argparse

    parser = argparse.ArgumentParser()
    add_parser(parser)
    args = parser.parse_args()

    model = make_parallel(args.num_proc, env_name, num=num, stochastic=True)
    env = make(env_name, num=num, resample_MP=True)

    args.iter_num = 3
    #args.horizon = 50
    args.num_mutation = 200
    args.num_elite = 20

    policy_net = POLO(value_net, model, action_space=env.action_space,
                            add_actions=args.add_actions,
                            horizon=args.horizon, std=args.std,
                            iter_num=args.iter_num,
                            initial_iter=args.initial_iter,
                            num_mutation=args.num_mutation, num_elite=args.num_elite, alpha=0.1, trunc_norm=True, lower_bound=env.action_space.low, upper_bound=env.action_space.high)

    env = make(env_name, num=num, resample_MP=True, stochastic=False)

    params = get_params(env)

    print("FIXXXXXXXXXXXXXXXXXXXXXXPARAMETERS")
    set_params(env, np.array([0.58093299, 0.05418986, 0.93399553, 0.1678795, 1.04150952]))
    set_params(env, [0.55111654,0.55281674,0.46355396,0.84531834,0.58944066])
    set_params(env, [0.31851129, 0.93941556, 0.02147825, 0.43523052, 1.02611646])

    mean_params = np.array([0.5] * len(params))
    osi = CEMOSI(model, mean_params,
        iter_num=10, num_mutation=100, num_elite=10, std=0.3)
    policy_net.set_params(mean_params)

    osi_eval(env, osi, policy_net, num_init_traj=2, max_horizon=20, eval_episodes=10, use_state=True, print_timestep=100, resample_MP=True)


if __name__ == '__main__':
    test_up_osi()
    #test_cem_osi()