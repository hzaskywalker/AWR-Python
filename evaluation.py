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
                print('\n\n', avg_reward, '\n\n')

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


def collect_trajectories(eval_env, policy, num_traj, max_horizon, use_state, use_done=True):
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
            action = policy(obs)
            if np.random.random() > 0: # explore
                action = eval_env.action_space.sample()
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


def osi_eval(eval_env, osi, policy, num_init_traj, max_horizon, eval_episodes, use_state=True, print_timestep=1000):


    resample_MP_init = eval_env.env.resample_MP
    rewards = []
    for episode in range(eval_episodes):
        osi.reset()

        eval_env.env.resample_MP = True
        eval_env.reset()
        eval_env.env.resample_MP = False

        for init_state, observations, actions, masks in collect_trajectories(eval_env, policy, num_init_traj, max_horizon, use_state):
            osi.update(init_state, observations, actions, masks)

        params = osi.get_params()
        policy.set_params(params)

        rewards += eval_policy(policy, eval_env, eval_episodes=1, use_state=use_state, set_gt_params=False)[1]

    mean, std = np.mean(rewards), np.std(rewards)
    print('mean, std', mean, std)

    eval_env.env.resample_MP = resample_MP_init
    return mean, std