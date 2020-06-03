# how to parallel awr?
# we do rollout in parallel, thus we need also parallelize the polices and synchronize the gradient..

import torch
import copy
import numpy as np
from .models import actor, critic
from robot.utils.normalizer import Normalizer
from robot.utils import soft_update, Timer
from collections import deque
import multiprocessing

# get the flat grads or params
def _get_flat_params_or_grads(network, mode='params'):
    """
    include two kinds: grads and params
    """
    attr = 'data' if mode == 'params' else 'grad'
    #return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])
    return torch.cat([getattr(param, attr).reshape(-1).cpu() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
    """
    include two kinds: grads and params
    """
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    for param in network.parameters():
        #getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        getattr(param, attr).copy_(flat_params[pointer:pointer + param.data.numel()].view_as(param.data))
        pointer += param.data.numel()


class AsyncDDPGAgent:
    def __init__(self, observation_space, action_space, temp=1., max_weight=20,
                 gamma=0.99, actor_lr=5e-5, critic_lr=0.01, tau=0.05, update_target_period=1, device='cpu',
                 batch_size=256, pipe=None):

        self.device = device
        inp_dim = observation_space.shape[0]
        self.actor = actor(inp_dim, action_space).to(self.device)
        self.critic = critic(inp_dim).to(self.device)
        self.normalizer = Normalizer((inp_dim,), default_clip_range=5).to(self.device)
        self.normalizer.count += 1
        self.temp = temp
        self.max_weight = max_weight

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), critic_lr)
        self.pipe = pipe
        self.batch_size = batch_size
        self.mse = nn.MSELoss()

    def set_params(self, params):
        assert isinstance(params, dict)
        _set_flat_params_or_grads(self.actor, params['actor'], mode='params'),
        _set_flat_params_or_grads(self.critic, params['critic'], mode='params')

    def get_params(self):
        return {
            'actor': _get_flat_params_or_grads(self.actor, mode='params'),
            'critic': _get_flat_params_or_grads(self.critic, mode='params')
        }


    def sync_grads(self, net, weights=1.):
        # reweight the gradients to avoid any bias...
        grad = _get_flat_params_or_grads(net, mode='grad')
        if weights != 1.:
            grad = grad * weights
        self.pipe.send(grad)
        grad = self.pipe.recv()
        _set_flat_params_or_grads(net, grad, mode='grad')


    def update_normalizer(self, obs):
        data= [obs.sum(axis=0), (obs**2).sum(axis=0), obs.shape[0]]
        self.pipe.send(data)
        s, sq, count = self.pipe.recv()
        self.normalizer.add(
            torch.tensor(s, dtype=torch.float32, device=self.device),
            torch.tensor(sq, dtype=torch.float32, device=self.device),
            torch.tensor(count, dtype=torch.long, device=self.device),
        )

    def tensor(self, x):
        return torch.tensor(x, dtype=torch.float).to(self.device)

    def as_state(self, s):
        return self.normalizer(self.tensor(s))

    def update_actor(self, steps, states, actions, advs):
        # we assume all numpy states is not normalized ...
        num_idx = len(states)
        for i in range(steps):
            self.optim_actor.zero_grad()
            idx = np.randon.sample(num_idx, self.batch_size)
            weights = np.minimum(np.exp(advs[idx]/self.temp), self.max_weight)
            weights = self.tensor(weights)
            logpi = -self.actor(self.as_state(states[idx])).log_probs(self.tensor(actions[idx]))
            assert logpi.shape == weights.shape
            actor_loss = (logpi * weights).mean()
            actor_loss.backward()

            self.sync_grads(self.actor)
            self.optim_actor.step()

    def update_critic(self, steps, states, targets):
        num_idx = len(states)
        for i in range(steps):
            self.optim_critic.zero_grad()

            idx = np.randon.sample(num_idx, self.batch_size)
            s = self.as_state(states[idx])
            v = self.tensor(targets[idx])
            assert s.shape == v.shape
            critic_loss = self.mse(s, v)
            critic_loss.backward()

            self.sync_grads(self.critic)
            self.optim_critic.step()

    def tocpu(self, x):
        return x.detach().cpu().numpy()

    def act(self, obs, mode='sample'):
        obs = self.as_state(obs)
        p = self.actor(obs)
        if mode == 'sample':
            a = p.sample()
        else:
            a = p.mean()
        return tocpu(a)

    def value(self, obs):
        return self.critic(self.as_state(obs))

    def update(self, buffer, actor_step, critic_step):
        states, actions, rewards, dones = [np.array(i) for i in buffer]
        value = self.critic(self.as_state(states))

    def discount_return(self, reward, done, value):
        value = value.squeeze()
        num_step = len(value)
        discounted_return = np.zeros([num_step])

        gae = 0
        for t in range(num_step - 1, -1, -1):
            if done[t]:
                delta = reward[t] - value[t]
            else:
                delta = reward[t] + gamma * value[t + 1] - value[t]
            gae = delta + gamma * lam * (1 - done[t]) * gae

            discounted_return[t] = gae + value[t]

        # For Actor
        adv = discounted_return - value
        return discounted_return, adv


class Worker(multiprocessing.Process):
    START = 1
    EXIT = 2
    ASK = 3
    GET_PARAM = 4
    SET_PARAM = 5

    def __init__(self, make, env_name,
                 replay_buffer_size=50000, sample_size=2048,
                 noise_eps=0.2, random_eps=0.3,
                 batch_size=256, future_K=4, seed=0, **kwargs):
        super(Worker, self).__init__()
        self.make = make
        self.env_name = env_name

        self.replay_buffer_size = replay_buffer_size
        self.sample_size = sample_size

        self.random_eps = random_eps
        self.batch_size = batch_size
        self.seed = seed

        self.kwargs = kwargs

        self.pipe, self.worker_pipe = multiprocessing.Pipe()
        self.start()

    def run(self):
        # initialize
        self.env = self.make(self.env_name)
        self.env.seed(self.seed)
        self.env.reset()
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        if '_max_episode_steps' in self.env.__dict__:
            self.T = self.env._max_episode_steps

        agent = AsyncDDPGAgent(self.env.observation_space, self.env.action_space, **self.kwargs, pipe=self.worker_pipe)
        agent.seed = self.seed

        states = deque(self.replay_buffer_size)
        actions = deque(self.replay_buffer_size)
        rewards = deque(self.replay_buffer_size)
        dones = deque(self.replay_buffer_size)
        episode = 0
        optim_iter = 0

        while True:
            op, data = self.worker_pipe.recv()
            if op == self.EXIT:
                break
            elif op == self.START:
                actor_step, critic_step, new_samples = data

                # new rollout
                obs = env.reset()
                episode_reward = 0
                num = 0

                # --------------- totally prallel ------------------
                while True:
                    action = agent.act(obs)
                    # feed the actions into the environment
                    states.append(np.array(obs))
                    obs, r, done, info = self.env.step(obs)

                    actions.append(action)
                    rewards.append(r)
                    dones.append(False)

                    num += 1
                    episode_reward += r
                    if done:
                        episode += 1
                        if num >= new_samples:
                            break
                        obs = env.reset()
                        episode_reward=0

                optim_iter += 1
                new_states = states[-num:] 
                if optim_iter == 1: #TODO: update 1
                    agent.update_normalizer(new_states)
                
                agent.update([states, actions, rewards, dones], actor_step, critic_step)

                if optim_iter != 1: # TODO: update 2
                    agent.update_normalizer(new_states)
                self.worker_pipe.send(None)

            elif op == self.ASK:
                action = agent.act(data)
                self.worker_pipe.send(action)
            elif op == self.GET_PARAM:
                self.worker_pipe.send(agent.get_params())
            elif op == self.SET_PARAM:
                agent.set_params(data)
            else:
                raise NotImplementedError

    def set_params(self, params):
        self.pipe.send([self.SET_PARAM, params])

    def get_params(self):
        self.pipe.send([self.GET_PARAM, None])
        return self.pipe.recv()

    def send(self, data):
        self.pipe.send(data)

    def recv(self):
        return self.pipe.recv()


class DDPGAgent:
    def __init__(self, n, num_epoch, n_rollout, timestep, n_batch=50, *args, recorder=None, seed=123, **kwargs):
        self.workers = []
        for i in range(n):
            self.workers.append(Worker(*args, timestep=timestep, **kwargs, seed=seed + i))
        self.recorder = recorder
        self.start(num_epoch, n_rollout, timestep, n_batch)

    def start(self, num_epoch, n_rollout, timestep, n_batch):
        primary = self.workers[0]
        params = primary.get_params()
        for i in self.workers:
            i.set_params(params)

        start_command = [primary.START, [n_rollout, timestep, n_batch]]
        for i in self.workers:
            i.send(start_command)
        for epoch_id in range(num_epoch):
            self.update_normalizer()

            for i in range(n_batch):
                self.reduce(mode='sum') # for critic
                self.reduce(mode='sum') # for actor

            for i in self.workers[1:]:
                i.recv(); i.send(start_command)

            # primary is special
            if self.recorder is not None:
                train_info = primary.recv()
                print(f"EPOCH {epoch_id}: REWARD {train_info[0]}")
                self.recorder.step(self, *train_info)
            primary.send(start_command)

    def __call__(self, observation):
        self.workers[0].send([self.workers[0].ASK, observation])
        return self.workers[0].recv()

    def reduce(self, mode='mean'):
        grad = self.workers[0].recv()
        for i in self.workers[1:]:
            grad = grad + i.recv()
        if mode == 'mean':
            grad = grad / len(self.workers)
        for i in self.workers:
            i.send(grad)

    def update_normalizer(self):
        s, sq, count, n = 0, 0, 0, len(self.workers)
        for i in self.workers:
            _s, _sq, _count = i.recv()
            s += _s; sq += _sq; count += _count
        s /= n; sq /= n; count /= n
        for i in self.workers:
            i.send([s, sq, count])


    def __del__(self):
        for i in self.workers:
            i.close()
