# finetune policy
import torch
import dill # for pickle


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


class Finetune:
    def __init__(self, env_name, num, agent, num_iter=21, num_proc=10):
        # pass
        self.num_iter = num_iter

        agent.optim_actor = None
        agent.optim_critic = None
        agent.normalizer.cpu()
        agent.critic.cpu()
        agent.actor.cpu()
        agent.device='cpu'
        agent.num = num
        self.env_name = env_name
        self.num = num
        self.num_proc = num_proc

        self.agent = agent
        self.awr = None

    def reset(self):
        pass

    def set_params(self, params):
        if len(params.shape) == 1:
            params = np.array([params])
        if self.awr is None:
            from envs import make
            from model import set_params
            from awr2.awr import AWR
            make2 = Maker(params[0], make, set_params, num=self.num)
            env_name = self.env_name

            actor_lr = 0.0001 if env_name[0]!='W' else 0.000025
            activation = 'tanh' if env_name[0] == 'W' else 'relu'

            self.awr = AWR(self.num_proc, make=make2, env_name=env_name, num_iter=0, device='cuda:0', replay_buffer_size=5000, path='tmp', optimizer='SGD', actor_lr = actor_lr, activation=activation)

        import copy
        agent2 = copy.deepcopy(self.agent)
        weights = clip_networks(agent2.normalizer, agent2.critic, agent2.actor, params.mean(axis=0))

        mean, std = params.mean(axis=0), params.std(axis=0)
        print('mean', mean, 'std', std)
        for idx, i in enumerate(self.awr.workers):
            i.copy(*weights)
            i.set_env(mean, std)
            i.reset()

        self.awr.start(num_iter=self.num_iter, new_samples=2048, critic_update_steps=20, actor_update_steps=200, sync_weights=False)

    def __call__(self, obs):
        return self.awr.workers[0].act(obs[None,:], mode='test')[0]