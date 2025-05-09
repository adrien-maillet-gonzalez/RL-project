import gym
import numpy as np
import torch
import random
from collections import deque
from TD3 import TD3  # Make sure TD3.py is in the same directory or adjust import
import torch.nn as nn

# Simple Replay Buffer
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = deque(maxlen=int(max_size))

    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(1 - done).unsqueeze(1)
        return state, action, next_state, reward, done


def evaluate_policy(env, policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def main():
    env_name = "MountainCarContinuous-v0"
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(state_dim, action_dim, max_action)

    replay_buffer = ReplayBuffer()

    total_timesteps = 200_000
    start_timesteps = 10_000  # Pure random actions to fill replay buffer
    expl_noise = 0.1
    batch_size = 256
    eval_freq = 5000

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(total_timesteps):
        episode_timesteps += 1

        # Select action with exploration noise
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(state))
            action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                env.action_space.low, env.action_space.high
            )

        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        replay_buffer.add((state, action, next_state, reward, done_bool))

        state = next_state
        episode_reward += reward

        # Train TD3 after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if done:
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate policy
        if (t + 1) % eval_freq == 0:
            evaluate_policy(env, policy)


if __name__ == "__main__":
    main()
