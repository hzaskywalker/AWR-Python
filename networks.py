import torch

class Value:
    def __init__(self, policy):
        self.policy = policy
    def __call__(self, state, params):
        state = torch.cat((state, params), dim=-1)
        act = self.policy.actor_target(state)
        cri1, cri2 = self.policy.critic_target(state,act)
        return torch.min(cri1, cri2)
        #return min(min (cri1.numpy(), cri2.numpy())[0][0])

def get_td3_value(env_name):
    if env_name == "DartWalker2dPT-v1":
        state_dim = 25
        action_dim = 6
        max_action = 1.0
    elif env_name == "DartHopperPT-v1":
        state_dim = 16
        action_dim = 3
        max_action = 1.0

    import utils
    import policy_transfer.uposi.TD3.utils
    from policy_transfer.uposi.TD3.TD3 import TD3
    import policy_transfer.uposi.TD3.OurDDPG
    import policy_transfer.uposi.TD3.DDPG

    policy = TD3(state_dim = state_dim, action_dim = action_dim, max_action = max_action)
    policy.load("/home/hza/policy_transfer/PT/policy_transfer/uposi/TD3/models/TD3_" + env_name + "_1000")
    #policy.actor_target.to(torch.device("cpu"))
    #policy.critic_target.to(torch.device("cpu"))
    policy.actor_target.to(torch.device("cuda"))
    policy.critic_target.to(torch.device("cuda"))
    return Value(policy)
