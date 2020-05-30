import tqdm
import gym
import numpy as np
from envs import make
from model import make_parallel, get_state, set_state, get_params

def test_env():
    env_name = 'DartHopperPT-v1'
    env = make(env_name, num=2)
    #env = gym.make('Walker2d-v2')
    #env.reset()

    for i in tqdm.trange(10000):
        env.step(env.action_space.sample())

def test_model():
    env_name = 'DartHopperPT-v1'
    env = make_parallel(1, env_name, num=2)

    env2 = make(env_name, num=2, stochastic=False)
    batch_size = 30
    horizon = 100
    
    s = []
    for i in range(batch_size):
        env2.reset()
        s.append(get_state(env2))

    param = get_params(env2)
    params = np.array([param for i in range(batch_size)])
    env2.env.noisy_input = False

    s = np.array(s)
    a = [[env2.action_space.sample() for j in range(horizon)] for i in range(batch_size)]
    a = np.array(a)

    for i in range(3):
        obs, _, done, _ = env2.step(a[-1][i])
        print(obs)
        if done:
            break


    for i in tqdm.trange(1):
        r, obs, mask = env(params, s, a)
    print(obs[-1][:3])

if __name__ == '__main__':
    #test_env()
    test_model()