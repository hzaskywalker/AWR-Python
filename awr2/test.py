import numpy as np
from awr import AWR
import torch
from gym import make
import sys
sys.path.append('/home/hza/policy_transfer/')
sys.path.append('/home/hza/policy_transfer/PT')

def test():
    """
    env_name = 'Hopper-v2'

    AWR(10, make=make, env_name=env_name, device='cuda:0', critic_update_steps=20, actor_update_steps=100, replay_buffer_size=5000)
    from gym import make
    AWR(10, make=make, env_name=env_name, device='cuda:0')
    """
    #env_name = 'InvertedPendulum-v2'
    #AWR(4, make=make, env_name=env_name, device='cuda:0', actor_lr=0.00005, action_std=0.2)

    #from envs import make
    #make2 = lambda env_name: make(env_name, num=5, resample_MP=True, stochastic=True, train_UP=True)
    #env_name = 'HopperPT-v2'
    #AWR(10, make=make2, env_name=env_name, device='cuda:0')

    from envs import make
    #env_name = 'HopperPT-v2'
    #env_name = 'HalfCheetahPT-v2'
    #env_name = 'Walker2dPT-v2'
    #AWR(10, make=make2, env_name=env_name, device='cuda:0', critic_update_steps=100, actor_update_steps=1000, replay_buffer_size=50000, path='HopperPT-v2')

    #env_name = 'Walker2dPT-v2'
    #AWR(10, make=make2, env_name=env_name, device='cuda:0', critic_update_steps=20, actor_update_steps=100, replay_buffer_size=5000, path='Walker2d-v2')

    env_name = 'HalfCheetahPT-v2'
    num = 6
    make2 = lambda env_name: make(env_name, num=num, resample_MP=True, stochastic=True, train_UP=True)
    AWR(1, make=make2, env_name=env_name, device='cuda:0', path='HalfCheetah2d-v2', actor_lr=0.0001, critic_lr=0.01, critic_update_steps=20, actor_update_steps=100, replay_buffer_size=5000)
    exit(0)

    env_name = 'Walker2dPT-v2'

    #num = 8
    num = 8
    make2 = lambda env_name: make(env_name, num=num, resample_MP=True, stochastic=True, train_UP=True)
    #def make3(env_name):
    #    env = make(env_name, num=num, resample_MP=True, stochastic=True, train_UP=True)
    #    from model import set_params
    #    set_params(env, np.array([1., 1., 1.]))
        
    #AWR(10, make=make2, env_name=env_name, device='cuda:0', path='Walker2d_2-v2', actor_lr=0.0001, critic_lr=0.01)
    #AWR(10, make=make2, env_name=env_name, device='cuda:0', path='Walker2d_2-v4', actor_lr=0.0001, critic_lr=0.01)
    #AWR(10, make=make2, env_name=env_name, device='cuda:0', path='Walker2d_2-v2', actor_lr=0.000025, critic_lr=0.01, critic_update_steps=20, actor_update_steps=100, replay_buffer_size=10000, activation='tanh', hidden_size=(128, 64), optimizer='SGD', action_std=0.4)
    AWR(10, make=make2, env_name=env_name, device='cuda:0', path='Walker2d_2-v3', actor_lr=0.000025, critic_lr=0.01, critic_update_steps=20, actor_update_steps=100, replay_buffer_size=50000, activation='tanh', hidden_size=(256, 128), optimizer='SGD', action_std=0.4)

class Maker:
    def __init__(self, params, make, set_params, num):
        self.params = params
        self.make = make
        self.set_params = set_params
        self.num = num

    def __call__(self, env_name):
        env = self.make(env_name, num=self.num, resample_MP=False, stochastic=True, train_UP=False)
        self.set_params(env, self.params)
        return env


def clip_networks(normalizer, critic, actor, params):
    num = params.shape[-1]

    normed_params = (torch.tensor(params, device=normalizer.sum.device).float() - normalizer.mean[-num:])/normalizer.std[-num:]
    normed_params = normed_params.clamp(-5, 5)
    #normed_params = torch.tensor(params, device='cuda:0').float()
    normalizer.size = (normalizer.size[0] - num,)
    normalizer.sum.data = normalizer.sum.data[:-num]
    normalizer.sumsq.data = normalizer.sumsq.data[:-num]
    normalizer.mean.data = normalizer.mean.data[:-num]
    normalizer.std.data = normalizer.std.data[:-num]

    critic.fc1.bias.data += (critic.fc1.weight[:, -num:] @ normed_params).data
    critic.fc1.weight.data = critic.fc1.weight.data[:, :-num]

    actor.fc1.bias.data += (actor.fc1.weight[:, -num:] @ normed_params).data
    actor.fc1.weight.data = actor.fc1.weight.data[:, :-num]
    return normalizer, critic, actor


def fine_tune():
    #env_name = 'HopperPT-v2'
    #num = 5
    env_name = 'Walker2dPT-v2'
    num = 8
    #env_name = 'HalfCheetahPT-v2'
    #num = 7

    from envs import make
    from model import get_params, set_params
    env = make(env_name, num=num, resample_MP=True, stochastic=True, train_UP=False)

    import torch
    import dill # for pickle
    from multiprocessing import set_start_method
    set_start_method('spawn')

    agent = torch.load(f'models/{env_name}')
    agent.normalizer.cpu()
    agent.critic.cpu()
    agent.actor.cpu()
    agent.device='cpu'
    agent.num = num

    #print(agent.act(np.concatenate((obs, params))[None,:], mode='test')[0])
    rewards = []
    rewards2 = []

    params = get_params(env)
    make2 = Maker(params, make, set_params, num=num)
    awr = AWR(1, make=make2, env_name=env_name, num_iter=0, device='cuda:0', replay_buffer_size=50000, path='tmp', optimizer='SGD', activation='tanh')
    print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
    print('activation for walker2d is tanh')

    from osi import seed
    for iter in range(30):
        reward = 0
        obs = env.reset()
        oo = obs.copy()
        params = get_params(env)
        print(params)
        while True:
            action = agent.act(np.concatenate((obs, params))[None,:], mode='test')[0]
            obs, r, done, _ = env.step(action)
            reward += r
            if done:
                break
        rewards.append(reward)
        print(iter, reward)

        import copy
        agent2 = copy.deepcopy(agent)
        weights = clip_networks(agent2.normalizer, agent2.critic, agent2.actor, params)

        """
        reward = 0
        obs = env.reset()
        while True:
            action = agent.act(obs[None,:], mode='sample')[0]
            obs, r, done, _ = env.step(action)
            reward += r
            if done:
                break
        #print(agent.act(oo[None,:], mode='test')[0])
        #print('second', reward)
        """
        #print('start copy')
        #agent2.actor.logstd.data[:] = np.log(0.4)
        for idx, i in enumerate(awr.workers):
            i.copy(*weights)
            i.set_env(params[idx%params.shape[0]])
            i.reset()

        #normalizer_dict, critic_dict, actor_dict = [i.state_dict() for i in weights]
        #agent2.critic.load_state_dict(critic_dict)
        #agent2.actor.load_state_dict(actor_dict)
        #agent2.normalizer.load_state_dict(normalizer_dict)
        #agent2.normalizer.size = (int(agent2.critic.fc1.weight.shape[-1]),)

        awr.start(num_iter=0, new_samples=2048, critic_update_steps=20, actor_update_steps=200, sync_weights=False)

        reward = 0
        obs = env.reset()
        set_params(env, params)

        while True:
            action = awr.workers[0].act(obs[None,:], mode='test')[0]
            #print(action)
            #action = agent2.act(obs[None,:], mode='test')[0]
            obs, r, done, _ = env.step(action)
            reward += r
            if done:
                break
        rewards2.append(reward)
        print('+', iter, reward)

    print(np.mean(rewards), np.mean(rewards2))


if __name__ == '__main__':
    #test()
    fine_tune()