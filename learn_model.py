# model learning baseline
# the goal is to fit a trajectory as good as possible ... 
import numpy as np
from evaluation import collect_trajectories


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

    # collect test trajectory
    test_traj = collect_trajectories(eval_env, policy, 10, max_horizon, use_state, use_done=True)

    # no online adapatation  
    loss = []
    for init_state, obs, actions, masks in test_traj:
        predict_obs, predict_mask = model.predict(init_state[None,:], obs[None, 0], actions[None,:])
        predict_obs = predict_obs[0]
        assert predict_obs.shape == obs.shape
        loss.append(
            (np.linalg.norm(obs - predict_obs) * masks[:, None])/masks[:, None].sum()
        )
    #print('mean loss', np.mean(loss))
    return np.mean(loss)

def evaluate(eval_env, policy, model, eval_episodes, num_init_traj, max_horizon, use_state=True):
    out = []
    for i in range(eval_episodes):
        out.append(evaluate_model_fit_(eval_env, policy, model, num_init_traj, max_horizon, use_state))
    print("Average loss: ", np.mean(out), " std:", np.std(out))
    return np.mean(out)

class OSIModel:
    # ensemble is a property of the osi
    def __init__(self, model, osi):
        self.osi = osi
        self.model = model

    def fit(self, trajectories):
        # fit the model with trajectoreis...
        for data in trajectories:
            self.osi.update(*data)
        self.params = self.osi.get_params()

    def fit2(self, trajectories):
        if self.online:
            self.fit(trajectories)

    def reset(self):
        self.osi.reset()

    def predict(self, init_state, init_obs, actions):
        param = np.tile(self.params[None,:], (len(init_state), 1))
        _, obs, mask =  self.model(param, init_state, actions)
        return obs, mask


def main():
    from osi import get_up_network, make, make_parallel, get_params, CEMOSI

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
    model = OSIModel(model, osi)

    for i in range(10):
        osi.cem.iter_num = i
        print(f'with {i} iteration')
        evaluate(env, policy_net, model, 10, 2, 15, use_state=False)


if __name__ == '__main__':
    main()