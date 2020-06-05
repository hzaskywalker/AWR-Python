import tqdm
import numpy as np
import cv2
from model import get_state, get_params, set_params

def eval_policy(policy, eval_env, eval_episodes=10, save_video=0, video_path="video{}.avi", timestep=int(1e9), use_state=True, set_gt_params=False, print_timestep=10000):

    avg_reward = 0.
    acc = []

    trajectories = []
    rewards = []
    for episode_id in tqdm.trange(eval_episodes):
        state, done = eval_env.reset(), False

        out = None
        if isinstance(policy, object):
            if 'reset' in policy.__dir__():
                policy.reset()

        if set_gt_params:
            policy.set_params(get_params(eval_env))

        #while not done:
        states = []
        actions = []
        for i in tqdm.trange(timestep):
            if i % print_timestep == print_timestep-1:
                print('\n\n', avg_reward, "past: ", rewards, '\n\n')

            if use_state:
                state = get_state(eval_env)
            states.append(state.tolist())
            action = policy(state)
            actions.append(action.tolist())
            state, reward, done, info = eval_env.step(action)
            avg_reward += reward
            if done:
                break
        states.append(state.tolist())

        if out is not None:
            out.release()
        trajectories.append([states, actions])

        rewards.append(avg_reward)
        avg_reward = 0


    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {np.mean(rewards):.3f},  std: {np.std(rewards)}")
    if len(acc) > 0:
        print(f"Evaluation success rate over {eval_episodes} episodes: {np.mean(acc):.3f}")
    print("---------------------------------------")
    return trajectories, rewards


def collect_trajectories(eval_env, policy, num_traj, max_horizon, use_state, use_done=True, random_policy=True):
    gt_params = get_params(eval_env)

    for i in range(num_traj):
        obs = eval_env.reset()
        set_params(eval_env, gt_params)

        init_state = get_state(eval_env)
        observations = None
        actions = None
        masks = None

        if use_state:
            obs = init_state

        policy.reset()
        for j in range(max_horizon):
            if np.random.random() > 0 and random_policy: # explore
                action = eval_env.action_space.sample()
            else:
                action = policy(obs)
            obs, _, done, _ = eval_env.step(action)

            if observations is None:
                observations = np.zeros((max_horizon, len(obs)))
                actions = np.zeros((max_horizon, len(action)))
                actions -= 10000000
                masks = np.zeros(max_horizon)

            observations[j] = obs
            actions[j] = action
            masks[j] = 1


            if use_state:
                # we always record the observation instead of the state
                obs = get_state(eval_env)
            if done and use_done:
                break

        if j == 0:
            continue

        yield init_state, observations, actions, masks


def osi_eval(eval_env, osi, policy, num_init_traj, max_horizon, eval_episodes, use_state=True, print_timestep=1000, resample_MP=True):


    resample_MP_init = eval_env.env.resample_MP
    rewards = []
    for episode in range(eval_episodes):
        osi.reset()

        eval_env.env.resample_MP = resample_MP
        eval_env.reset()
        eval_env.env.resample_MP = False

        for init_state, observations, actions, masks in collect_trajectories(eval_env, policy, num_init_traj, max_horizon, use_state):
            osi.update(init_state, observations, actions, masks)

        params = osi.get_params()
        print('find params', params)
        policy.set_params(params)

        rewards += eval_policy(policy, eval_env, eval_episodes=1, use_state=use_state, set_gt_params=False, timestep=1000, print_timestep=print_timestep)[1]
        print(get_params(eval_env), params)

    mean, std = np.mean(rewards), np.std(rewards)
    print('mean, std', mean, std)

    eval_env.env.resample_MP = resample_MP_init
    return mean, std

def online_osi(eval_env, osi, policy, num_init_traj, max_horizon, eval_episodes, use_state=True, print_timestep=1000, resample_MP=True, online=True, ensemble=1, gt=False):
    # fix the seed...
    from osi import seed 
    seed(eval_env, 0)
    parameters = []
    for i in range(100):
        eval_env.reset()
        parameters.append(get_params(eval_env))

    resample_MP_init = eval_env.env.resample_MP
    rewards = []
    for episode in range(eval_episodes):
        osi.reset()

        eval_env.env.resample_MP = resample_MP
        eval_env.reset()
        eval_env.env.resample_MP = False
        if parameters is not None:
            set_params(eval_env, parameters[episode])

        for init_state, observations, actions, masks in collect_trajectories(eval_env, policy, num_init_traj, max_horizon, use_state):
            osi.update(init_state, observations, actions, masks)

        #params = osi.get_params()
        if gt:
            params = get_params(eval_env)
        else:
            params = osi.find_min(ensemble, method='all') # get a good initialization
        policy.set_params(params)
        print(params, get_params(eval_env))

        reward = 0
        obs, state = eval_env.reset(), get_state(eval_env)
        policy.reset()


        states = []
        observations = []
        actions = []
        states.append(states)

        for i in tqdm.trange(1000):
            if use_state:
                action = policy(state)
            else:
                action = policy(obs)

            obs, r, done, _ = eval_env.step(action)
            state = get_state(eval_env)
            states.append(state)


            observations.append(obs)
            actions.append(action)

            if i % print_timestep == print_timestep - 1:
                print('\n\n', reward, "past: ", rewards[-10:], len(rewards), '\n\n')

            if i % max_horizon == max_horizon - 1 and i > max_horizon + 3 and online:
                xx = i//max_horizon
                if xx % online == online - 1:
                    idx = i - max_horizon - 1
                    osi.update(states[idx], observations[idx:idx+max_horizon], actions[idx:idx+max_horizon], 1, maxlen=3)
                    tmp = osi.cem.iter_num
                    #osi.cem.iter_num = 5 # we need at least 10 iterations??
                    osi.cem.iter_num = 10 # we need at least 10 iterations??
                    osi.cem.std = 0.1
                    osi.cem.num_mutation = 100
                    osi.cem.num_elite = 5
                    params = params * 0.5 + osi.get_params() * 0.5 # don't know if this is ok
                    policy.set_params(params)
                print(params, get_params(eval_env))
                print('\n\n', reward, "past: ", rewards[-10:], len(rewards), '\n\n')


            reward += r
            #if i % print_timestep == print_timestep-1 or done:
            #    print('\n\n', reward, "past: ", rewards[-10:], len(rewards), '\n\n')
            if done:
                break
        rewards.append(reward)


    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {np.mean(rewards):.3f},  std: {np.std(rewards)}")
    print("---------------------------------------")
    return rewards