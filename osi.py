# different osi model
# osi model have the following form:
#   init(trajectories)
#   update()
import numpy as np
from optim import CEM
import torch
from collections import deque
from evaluation import osi_eval, online_osi
from networks import get_up_network
from model import make_parallel, make, get_params, set_params


class CEMOSI:
    def __init__(self, model, init_mean, iter_num, num_mutation, num_elite,
        *args, std=0.3, ensemble_num=1, queue_size=5, **kwargs):
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
        self.ensemble_num = ensemble_num

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
        norm = np.linalg.norm(norm, axis=-1)
        error = norm.sum(axis=(1, 2)) 

        if negative_mask.any():
            error[negative_mask] = 1e9
        return torch.tensor(error, dtype=torch.float, device=device)

    def update(self, state, obs, action, mask, maxlen=20):
        self.states.append(state)
        self.actions.append(action)
        self.obs.append(obs)
        self.masks.append(mask)

        maxlen = min(maxlen, len(self.states))
        self.states = self.states[-maxlen:]
        self.actions = self.actions[-maxlen:]
        self.obs = self.obs[-maxlen:]
        self.masks = self.masks[-maxlen:]
    
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

    def get_params(self, ensemble_num=None):
        # optim
        if ensemble_num is None:
            ensemble_num = self.ensemble_num
        if ensemble_num != 1:
            return self.find_min(self.ensemble_num, method='all')
        self.optim()
        return self._params.clip(-0.05, np.inf)

    def find_min(self, ensemble_num, method='min'):
        min_params = None
        loss = 0
        params = []

        pp = 0
        for i in range(ensemble_num):
            self._params= np.random.random(size=self.init.shape) * 0.6 + 0.2
            out = self.get_params(1)
            if self._loss < loss or min_params is None:
                loss = self._loss
                min_params = out
            pp = pp + out
            params.append(out)
        #return params
        if method == 'all':
            return np.stack(params)
        elif method == 'average':
            return pp/ensemble_num
        elif method == 'min':
            return min_params
        else:
            raise NotImplementedError


class DiffOSI(CEMOSI):
    def __init__(self, model, init, lr=0.01, eps=1e-4, iter=30, momentum=0.9):
        self.model = model
        self.iters = iter
        self.eps = eps
        self.momentum = momentum
        self.lr = lr
        self.init = init

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
        norm = np.linalg.norm(norm, axis=-1)
        error = norm.sum(axis=(1, 2)) 

        if negative_mask.any():
            error[negative_mask] = 1e9
        return torch.tensor(error, dtype=torch.float, device=device)

    def optim(self):
        mean = self._params
        momentum = 0
        self.lr = 0.001
        for i in range(self.iters):
            haha = []
            for j in range(len(self._params)):
                x = mean.copy()
                x[j] += self.eps
                haha.append(x)
                x = mean.copy()
                x[j] -= self.eps
                haha.append(x)

            inp = torch.tensor(np.array(haha), dtype=torch.float, device='cuda:0')
            cost = self.cost(None, inp) 
            grad = cost.reshape(-1, 2)
            grad = (grad[:, 0] - grad[:, 1])/self.eps/2
            grad = grad.detach().cpu().numpy()

            #momentum = momentum * self.momentum + grad * (1-self.momentum)
            #mean = mean - momentum * self.lr
            mean = mean - grad * self.lr

            mean = mean.clip(-0.05, np.inf)
        self._params = mean


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


    online_osi(env, osi, policy_net, num_init_traj=1, max_horizon=15, eval_episodes=50, use_state=False, print_timestep=10000, resample_MP=True)


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

    #args.iter_num = 2
    #args.num_mutation = 200
    args.num_mutation = 100
    args.iter_num = 2
    args.num_elite = 10

    policy_net = POLO(value_net, model, action_space=env.action_space,
                            add_actions=args.add_actions,
                            horizon=args.horizon, std=args.std,
                            iter_num=args.iter_num,
                            initial_iter=args.initial_iter,
                            num_mutation=args.num_mutation, num_elite=args.num_elite, alpha=0.1, trunc_norm=True, lower_bound=env.action_space.low, upper_bound=env.action_space.high)

    resample_MP = True
    env = make(env_name, num=num, resample_MP=resample_MP, stochastic=False)

    params = get_params(env)

    print("FIXXXXXXXXXXXXXXXXXXXXXXPARAMETERS")
    set_params(env, np.array([0.58093299, 0.05418986, 0.93399553, 0.1678795, 1.04150952]))
    set_params(env, [0.55111654,0.55281674,0.46355396,0.84531834,0.58944066])
    set_params(env, [0.31851129, 0.93941556, 0.02147825, 0.43523052, 1.02611646])
    set_params(env, [0.58589476, 0.11078934, 0.348238, 0.68130195, 0.98376274])

    mean_params = np.array([0.5] * len(params))
    osi = CEMOSI(model, mean_params,
        iter_num=20, num_mutation=100, num_elite=10, std=0.3, ensemble_num=5)
    policy_net.set_params(mean_params)
    print(get_params(env))

    online_osi(env, osi, policy_net, num_init_traj=1, max_horizon=15, eval_episodes=10, use_state=True, print_timestep=10, resample_MP=resample_MP)


def test_up_diff():
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
    #osi = DiffOSI(model, mean_params, 0.001, iter=100, momentum=0.9, eps=1e-5)
    policy_net.set_params(mean_params)


    # I run this at the last time..
    # online is very useful ..
    online_osi(env, osi, policy_net, num_init_traj=1, max_horizon=15, eval_episodes=10, use_state=False, print_timestep=10000, resample_MP=True)
    #osi_eval(env, osi, policy_net, num_init_traj=1, max_horizon=100, eval_episodes=10, use_state=False, print_timestep=10000, resample_MP=True)



if __name__ == '__main__':
    #test_up_osi()
    #test_cem_osi()
    test_up_diff()