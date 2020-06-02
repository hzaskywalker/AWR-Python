# policy has the following api
#   - reset
#   - set param: update the current understanding about the environment
#   - __call__: return the action
import torch
import numpy as np
import argparse
from optim import RolloutCEM
from evaluation import eval_policy
from networks import get_td3_value, get_up_network
from model import make_parallel, make, get_params


class POLO(RolloutCEM):
    def __init__(self, value_net=None, model=None, *args, **kwargs):
        model.rollout = None
        super(POLO, self).__init__(model, *args, **kwargs)
        self.params = None
        self.value_net = value_net

    def set_params(self, params):
        if isinstance(params, np.ndarray):
            params = torch.tensor(params, device='cuda:0', dtype=torch.float32)
        self.params = params # parameters

    def cost(self, x, a):
        assert self.params is not None, "Please set the parameters first"

        if len(self.params.shape) == 2:
            # ensemble..
            costs = []
            pp = self.params
            for j in self.params:
                self.params = j
                costs.append(self.cost(x, a))
            self.params = pp
            out = torch.stack(costs).min(dim=0)[0]
            return out

        device = x.device
        dtype = x.dtype

        params = self.params
        if params.dim() == 1:
            params = params[None,:].expand(a.shape[0], -1)


        x = x[None, :].expand(a.shape[0], -1).detach().cpu().numpy()
        a = a.detach().cpu().numpy()
        r, obs, mask = self.model(params.detach().cpu().numpy(), x, a)
        r = torch.tensor(r, dtype=torch.float32, device=device) # original reward..
        mask = torch.tensor(mask, device='cuda:0', dtype=torch.float)

        if self.value_net is not None:
            params = params[:, None].expand(-1, obs.shape[1], -1)
            params = params.reshape(-1, params.shape[-1])
            obs = torch.tensor(obs.reshape(-1, obs.shape[-1]), device='cuda:0', dtype=torch.float)

            values = self.value_net(obs, params).reshape(*a.shape[:2])
            #r = r + (values * mask).sum(dim=-1) * 0.0
            #r = r
            r = r + (values * mask).sum(dim=-1) * 0.01
        else:
            r -= ((1-mask) * 2).sum(dim=-1)

        return -r


def add_parser(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='DartHopperPT-v1')
    parser.add_argument('--iter_num', type=int, default=2)
    parser.add_argument('--initial_iter', type=int, default=0)
    parser.add_argument('--num_mutation', type=int, default=100)
    parser.add_argument('--num_elite', type=int, default=5)
    parser.add_argument('--std', type=float, default=0.5)
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--num_proc', type=int, default=30)
    parser.add_argument('--video_num', type=int, default=1)
    parser.add_argument('--video_path', type=str, default='video{}.avi')
    parser.add_argument('--output_path', type=str, default='tmp.json')
    parser.add_argument('--num_test', type=int, default=1)
    parser.add_argument('--timestep', type=int, default=1000)
    parser.add_argument('--add_actions', type=int, default=1)
    parser.add_argument('--controller', type=str, default='cem', choices=['cem', 'poplin'])


def test_POLO():
    #env_name = 'DartWalker2dPT-v1'
    #num = 8
    env_name = 'DartHopperPT-v1'
    num = 5

    value_net = get_td3_value(env_name)
    #value_net = None

    parser = argparse.ArgumentParser()
    add_parser(parser)
    args = parser.parse_args()


    model = make_parallel(args.num_proc, env_name, num=num, stochastic=True)
    env = make(env_name, num=num, resample_MP=False)

    controller = POLO(value_net, model, action_space=env.action_space,
                            add_actions=args.add_actions,
                            horizon=args.horizon, std=args.std,
                            iter_num=args.iter_num,
                            initial_iter=args.initial_iter,
                            num_mutation=args.num_mutation, num_elite=args.num_elite, alpha=0.1, trunc_norm=True, lower_bound=env.action_space.low, upper_bound=env.action_space.high)

    trajectories = eval_policy(controller, env, 10, args.video_num, args.video_path, timestep=args.timestep, set_gt_params=True, print_timestep=100)

def test_UP():
    env_name = 'DartHopperPT-v1'
    num = 5

    policy_net = get_up_network(env_name, num)

    env = make(env_name, num=num, resample_MP=True)
    eval_policy(policy_net, env, 10, 0, None, timestep=1000, use_state=False, set_gt_params=True)


if __name__ == '__main__':
    test_POLO()
    #test_UP()