from awr import AWR
from gym import make

def test():
    env_name = 'Hopper-v2'
    AWR(1, make=make, env_name=env_name, device='cuda:0')
    #env_name = 'InvertedPendulum-v2'
    #AWR(4, make=make, env_name=env_name, device='cuda:0', actor_lr=0.00005, action_std=0.2)

if __name__ == '__main__':
    test()