# model learning baseline
# the goal is to fit a trajectory as good as possible ... 
import numpy as np
from evaluation import collect_trajectories
from model import get_params


def cost(model, test_traj):
    loss = []
    for init_state, obs, actions, masks in test_traj:
        predict_obs, predict_mask = model.predict(init_state[None,:], obs[None, 0], actions[None,:])
        predict_obs = predict_obs[0]
        assert predict_obs.shape == obs.shape
        loss.append(
            (np.linalg.norm(obs - predict_obs) * masks[:, None])/masks[:, None].sum()
        )
    return np.mean(loss)

def evaluate_model_fit_(eval_env, policy, model, num_init_traj, max_horizon, use_state=True):
    """
    Model fit experiments ...
    FIrst 
    """
    eval_env.env.resample_MP = True
    eval_env.reset()
    eval_env.env.resample_MP = False
    model.reset()

    # osi will return the parameters

    trajs = []
    for data in collect_trajectories(eval_env, policy, num_init_traj, max_horizon, use_state):
        trajs.append(data)
    model.fit(trajs)
    #print('gt', get_params(eval_env))


    # collect test trajectory
    train = cost(model, trajs)
    test_traj = collect_trajectories(eval_env, policy, 10, max_horizon, use_state, use_done=True)
    test = cost(model, test_traj)
    #print('loss', test, 'train', train)
    # no online adapatation  
    #print('mean loss', np.mean(loss))
    return train, test

def evaluate(eval_env, policy, model, eval_episodes, num_init_traj, max_horizon, use_state=True):
    out_train, out_test = [], []
    for i in range(eval_episodes):
        train, test = evaluate_model_fit_(eval_env, policy, model, num_init_traj, max_horizon, use_state)
        out_train.append(train)
        out_test.append(test)
    print("Average train loss: ", np.mean(out_train), " std:", np.std(out_train))
    print("Average test loss: ", np.mean(out_test), " std:", np.std(out_test))
    return np.mean(out_test)

class OSIModel:
    # ensemble is a property of the osi
    def __init__(self, model, osi, ensemble=1):
        self.osi = osi
        self.model = model
        self.ensemble = ensemble

    def fit(self, trajectories):
        # fit the model with trajectoreis...
        for data in trajectories:
            self.osi.update(*data)
        if self.ensemble == 1:
            self.params = self.osi.get_params().clip(-0.05, np.inf) # clamp to avoid error ..
        else:
            # choose min as the ensemble
            #self.params = [ for i in range(self.ensemble)]
            params = self.osi.find_min(self.ensemble).clip(-0.05, np.inf)
            self.params = params
            #print('result', self.params, 'loss:', loss)

    def reset(self):
        self.osi.reset()

    def predict(self, init_state, init_obs, actions):
        def predict(param):
            param = np.tile(param[None,:], (len(init_state), 1))
            _, obs, mask =  self.model(param, init_state, actions)
            return obs, mask
        #if self.ensemble == 1 or:
        return predict(self.params)
        """
        else:
            obs = []
            for i in self.params:
                o, mask = predict(i)
                obs.append(o)
            return np.mean(np.array(obs), axis=0), mask
            """


from osi import get_up_network, make, make_parallel, get_params, CEMOSI
def main():

    env_name = 'DartHopperPT-v1'
    num = 5
    policy_net = get_up_network(env_name, num)


    model = make_parallel(10, env_name, num=num, stochastic=False, done=False)
    env = make(env_name, num=num, resample_MP=True, stochastic=False)

    params = get_params(env)
    mean_params = np.array([0.5] * len(params))

    osi = CEMOSI(model, mean_params,
        iter_num=1, num_mutation=100, num_elite=10, std=0.3)
    policy_net.set_params(mean_params)

    for ensemble_num in range(5, 6):
        haha = OSIModel(model, osi, ensemble=ensemble_num)
        osi.cem.iter_num = 10
        evaluate(env, policy_net, haha, 30, 1, 15, use_state=False)
        print(f'with {ensemble_num} ensemble')


from dataset import Dataset

def learn_with_dataset():

    env_name = 'DartHopperPT-v1'
    num = 5

    env = make(env_name, num)

    dataset = Dataset(env_name, num)
    params = get_params(env)
    mean_params = np.array([0.5] * len(params))

    model = make_parallel(10, env_name, num=num, stochastic=False, done=False)
    osi = CEMOSI(model, mean_params,
        iter_num=1, num_mutation=100, num_elite=10, std=0.3)

    learner = OSIModel(model, osi, ensemble=3)
    osi.cem.iter_num = 10

    for test, train, train_online, params in zip(*dataset.data):
        learner.reset()
        trajs = [[train[j][i] for j in range(3)] + [np.ones((1,))]for i in range(train[0].shape[0])]
        learner.fit(trajs)
        print('===========')
        #print(learner.osi.get_params(), params)
        print(cost(learner, [list(test)+[np.ones((1,))]]))

        learner.reset()
        trajs = [[train_online[j][i] for j in range(3)] + [np.ones((1,))]for i in range(train_online[0].shape[0])]
        learner.fit(trajs)
        #print(learner.osi.get_params(), params)
        print(cost(learner, [list(test)+[np.ones((1,))]]))

if __name__ == '__main__':
    #main()
    learn_with_dataset()