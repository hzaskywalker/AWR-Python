import tqdm
import cv2
from model import get_state

def eval_policy(policy, eval_env, eval_episodes=10, save_video=0, video_path="video{}.avi", timestep=int(1e9)):

    avg_reward = 0.
    acc = []

    trajectories = []
    eval_episodes=1
    for episode_id in tqdm.trange(eval_episodes):
        state, done = eval_env.reset(), False

        out = None
        if isinstance(policy, object):
            if 'reset' in policy.__dir__():
                policy.reset()

        #while not done:
        states = []
        actions = []
        for i in tqdm.trange(timestep):
            if i % 1 == 0:
                print('\n\n', avg_reward, '\n\n')

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

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    if len(acc) > 0:
        print(f"Evaluation success rate over {eval_episodes} episodes: {np.mean(acc):.3f}")
    print("---------------------------------------")
    return trajectories