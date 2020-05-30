import gym
import sys
sys.path.append('PT/')
import policy_transfer.envs


def make(env_name, num, seed = 1000, stochastic=True):
    env = None
    if env_name == "DartHopperPT-v1" and num == 2:
        env = make_env(env_name, [0,5], seed)
    if env_name == "DartHopperPT-v1" and num == 5:
        env = make_env(env_name, [0,1,2,5,9], seed)
    if env_name == "DartHopperPT-v1" and num == 10:
        env = make_env(env_name, [0,1,2,3,4,5, 6,7,8, 9], seed)
    if env_name == "DartWalker2dPT-v1" and num == 8:
        env = make_env(env_name, [7,8,9,10,11,12, 13,14], seed)
    if env_name == "DartWalker2dPT-v1" and num == 15:
        env = make_env(env_name, [0,1,2,3,4,5,6,7,8,9,10,11,12, 13,14], seed)
    if env_name == "DartHalfCheetahPT-v1" and num == 8:
        env = make_env(env_name, [0,1,2,3,4,5,6,7], seed)
    if not stochastic:
        env.env.noisy_input = stochastic
    return env
    

def make_env(env_name, dyn_params, seed=1000):
    env = gym.make(env_name)
    env.seed(seed)
    env.env.param_manager.activated_param = dyn_params
    env.env.param_manager.controllable_param = dyn_params
    env.reset()
    return env

if __name__ == '__main__':
    env = make("DartWalker2dPT-v1", 15)
    print(env.env.param_manager.get_simulator_parameters())
    env = make("DartHopperPT-v1", 10)
    print(env.env.param_manager.get_simulator_parameters())