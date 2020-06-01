import os
import tqdm
import pickle
import numpy as np
from networks import get_up_network 
from envs import make
from model import get_params, get_state

class Dataset:
    def __init__(self, env_name, num, total=20, num_train=15, max_horizon=15):
        # pass
        self.env_name = env_name
        self.num = num
        self.total = total
        self.num_train = num_train
        self.max_horizon = max_horizon

        self.policy = get_up_network(env_name, num)

        path = f"{env_name}_{num}"

        self.eval_env = make(env_name, num)
        self.data = self._get_data(path)

    def _get_data(self, path):
        if os.path.exists(path): 
            with open(path, 'rb') as f:
                return pickle.load(f)


        gt_params = []

        train_traj_offline = [] # trajs at random position..
        train_traj_online = [] # recent trajs
        test_traj = []

        eval_env = self.eval_env
        eval_env.env.resample_MP = False

        for i in tqdm.trange(self.total):
            eval_env.env.resample_MP = True
            eval_env.reset()
            eval_env.env.resample_MP = False
            gt_param = get_params(eval_env)
            gt_params.append(gt_param)
            self.policy.set_params(gt_param)
            self.policy.reset()

            # collect_train_traj 
            states = []
            observations = []
            actions = []

            obs = eval_env.reset()

            while True:
                # collect the whole trajectories
                # policy 1
                action = self.policy(obs)
                states.append(get_state(eval_env))
                actions.append(action)

                obs, r, done, _ = eval_env.step(action)
                observations.append(obs)
                if done:
                    break

            if len(observations) < self.max_horizon * 2:
                continue

            # traj: states, observation, actions, mask
            test_idx = np.random.randint(self.max_horizon, len(states)-self.max_horizon)
            test_traj.append((
                states[test_idx], np.array(observations[test_idx:test_idx + self.max_horizon]), np.array(actions[test_idx:test_idx + self.max_horizon])
            ))

            train = []
            train_online = []
            #for i in range(self.num_train):
                #idx = np.random.randint(len(states))
            for idx in range(self.num_train):
                train.append((states[idx], observations[idx:idx+1], np.array(actions[idx:idx+1])))

            #for idx in range(test_idx - self.max_horizon, test_idx-1):
            for i in range(self.num_train):
                idx = np.random.randint(test_idx)
                train_online.append((states[idx], observations[idx:idx+1], actions[idx:idx+1]))
            train = [np.array([j[i] for j in train]) for i in range(3)]
            train_online = [np.array([j[i] for j in train_online]) for i in range(3)]

            train_traj_offline.append(train)
            train_traj_online.append(train_online)

        print(np.array(gt_params).shape)
        data = [test_traj, train_traj_offline, train_traj_online, gt_params] 
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return data



def main():
    env_name = 'DartHopperPT-v1'
    num = 5

    dataset = Dataset(env_name, num, 20, num_train=15, max_horizon=15)


if __name__ == '__main__':
    main()