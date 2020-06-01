from envs import make
import numpy as np
from model import get_state, set_state
import tqdm 

def test():
    env = make('DartHopperPT-v1', num=5)
    """
    env.reset()
    for i in tqdm.trange(10000):
        env.step(env.action_space.sample())
        """

    env.reset()
    state = get_state(env)
    for i in tqdm.trange(10000):
        env.reset()
        set_state(env, state)
        state = state + np.random.normal(state.shape)
        env.step(env.action_space.sample())
        state = get_state(env)

if __name__ == '__main__':
    test()