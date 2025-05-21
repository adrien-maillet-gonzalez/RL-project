import os
import random
import numpy as np
import tensorflow as tf
import functools
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import trange

from tf_agents.environments import suite_gym, ParallelPyEnvironment, tf_py_environment
from tf_agents.agents.sac.sac_agent import SacAgent
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks import sequential, nest_map
from tf_agents.keras_layers import inner_reshape
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.system import multiprocessing

try:
    multiprocessing.enable_interactive_mode()
except RuntimeError as e:
    if "context has already been set" not in str(e):
        raise
except ValueError as e:
    if "Multiprocessing already initialized" not in str(e):
        raise

# Habilitar uso de GPU
physical_devices = tf.config.list_physical_devices('GPU')
for g in physical_devices:
    tf.config.experimental.set_memory_growth(g, True)
print("Usando GPU:", tf.config.list_logical_devices('GPU'))

# --- Helper para construir red critic personalizada ---
dense = functools.partial(tf.keras.layers.Dense, activation='relu', kernel_initializer='glorot_uniform')

def create_identity_layer():
    return tf.keras.layers.Lambda(lambda x: x)

def create_sequential_critic_network(obs_units, act_units, joint_units):
    def split(inputs):
        return {'observation': inputs[0], 'action': inputs[1]}
    obs_net   = sequential.Sequential([dense(u) for u in obs_units]) if obs_units else create_identity_layer()
    act_net   = sequential.Sequential([dense(u) for u in act_units]) if act_units else create_identity_layer()
    joint_net = sequential.Sequential([dense(u) for u in joint_units]) if joint_units else create_identity_layer()
    value_layer = tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')
    return sequential.Sequential([
        tf.keras.layers.Lambda(split),
        nest_map.NestMap({'observation': obs_net, 'action': act_net}),
        nest_map.NestFlatten(),
        tf.keras.layers.Concatenate(),
        joint_net,
        value_layer,
        inner_reshape.InnerReshape(current_shape=[1], new_shape=[])
    ], name='sequential_critic')


def run_sac_seed(seed,
                 #env_name = "MountainCarContinuous-v0",
                 env_name="Pendulum-v1",
                 num_parallel=200,
                 collect_steps=200,
                 batch_size=256*2,
                 replay_buffer_max=200_000,
                 learning_rate=1e-4,
                 num_iterations=100_000,#50_000,
                 eval_interval=10_000):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

    def make_env(): return suite_gym.load(env_name)
    py_env = ParallelPyEnvironment([make_env] * num_parallel)
    train_env = tf_py_environment.TFPyEnvironment(py_env)
    eval_env  = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

    with tf.device('/GPU:0'):
        train_step = tf.Variable(0)

        actor_net  = ActorDistributionNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=(256,256)
        )
        critic_net1 = create_sequential_critic_network((256,256), None, (256,256))
        critic_net2 = create_sequential_critic_network((256,256), None, (256,256))

        agent = SacAgent(
            time_step_spec=train_env.time_step_spec(),
            action_spec=train_env.action_spec(),
            actor_network=actor_net,
            critic_network=critic_net1,
            critic_network_2=critic_net2,
            actor_optimizer=tf.keras.optimizers.Adam(learning_rate),
            critic_optimizer=tf.keras.optimizers.Adam(learning_rate),
            alpha_optimizer=tf.keras.optimizers.Adam(learning_rate),
            target_update_tau=0.005,
            target_update_period=1,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=0.99,
            reward_scale_factor=2.0,
            train_step_counter=train_step
        )
        agent.initialize()
        agent.train = common.function(agent.train)

    buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=num_parallel,
        max_length=replay_buffer_max
    )
    dataset = buffer.as_dataset(sample_batch_size=batch_size, num_steps=2).prefetch(tf.data.AUTOTUNE)
    iterator = iter(dataset)

    driver = dynamic_step_driver.DynamicStepDriver(
        train_env, agent.collect_policy, observers=[buffer.add_batch], num_steps=collect_steps
    )
    driver.run()  # Warm-up

    @tf.function
    def train_step_fn():
        experience, _ = next(iterator)
        return agent.train(experience)

    episodes = []
    evals = []
    ep_rewards = np.zeros(num_parallel)
    ep_steps = np.zeros(num_parallel, dtype=int)
    ep_count = np.zeros(num_parallel, dtype=int)

    def update_episodes(time_step):
        nonlocal ep_rewards, ep_steps, ep_count
        rewards = time_step.reward.numpy()
        dones = time_step.is_last().numpy()
        ep_rewards += rewards
        ep_steps += 1
        for i, done in enumerate(dones):
            if done:
                ep_count[i] += 1
                episodes.append({
                    "total_timesteps": int(ep_steps[i]),
                    "episode_num": int(ep_count[i]),
                    "episode_timesteps": int(ep_steps[i]),
                    "reward": float(ep_rewards[i])
                })
                ep_rewards[i] = 0.0
                ep_steps[i] = 0
    start_time = datetime.now().isoformat(timespec='seconds')

    pbar = trange(num_iterations + 1, desc=f"Seed {seed}", dynamic_ncols=True)
    pbar.set_postfix({"eval_return": "N/A"})
    for step in pbar:
        time_step, _ = driver.run()
        update_episodes(time_step)
        train_step_fn()

        if step % eval_interval == 0:
            ts = eval_env.reset(); total = 0.0
            while not ts.is_last():
                action_step = agent.policy.action(ts)
                ts = eval_env.step(action_step.action)
                total += ts.reward.numpy().item()
            evals.append({
                "at_timesteps": int(step),
                "evaluation_over_1_episode": float(total)
            })
            print(f"[Step {step:>5}] Eval return = {total:.2f}")
            pbar.set_postfix({"eval_return": f"{total:.2f}"})
            pbar.update(0)

    data = {
        "experiment": {
            "policy": "SAC",
            "environment": env_name,
            "seed": seed,
            "start_time": start_time
        },
        "episodes": episodes,
        "evaluations": evals
    }

    folder = "jsons"
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    fname = f"sac_{seed}_{ts}.json"
    path = os.path.join(folder, fname)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON to {path}")

    return episodes, evals, agent


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

seeds = [0, 1, 2]
all_rewards = []

for s in seeds:
    eps, evs, agent = run_sac_seed(seed=s)

    # Store episode rewards
    rewards = [e['reward'] for e in eps]
    all_rewards.append(rewards)

# Find the minimum common length (in case training was cut short)
min_len = min(len(r) for r in all_rewards)
all_rewards = [r[:min_len] for r in all_rewards]

# Convert to numpy array for easier math
reward_array = np.array(all_rewards)  # shape: (seeds, episodes)
mean_rewards = np.mean(reward_array, axis=0)
std_rewards = np.std(reward_array, axis=0)

episodes = np.arange(min_len)

# Linear trend line
slope, intercept, *_ = linregress(episodes, mean_rewards)
trend_line = slope * episodes + intercept

# Plot: Mean ± Std Dev and Trend
plt.figure(figsize=(10, 6))
plt.plot(episodes, mean_rewards, color='blue', label='Mean Reward')
plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, color='blue', alpha=0.2, label='±1 Std Dev')
plt.plot(episodes, trend_line, 'r--', label=f"Trend: {slope:.2f}x + {intercept:.2f}")

plt.title('Mean Episode Rewards Across Seeds - SAC on MountainCarContinuous-v0')
plt.xlabel('Episode Number')
plt.ylabel('Mean Total Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sac_rewards_plotMountaincar.png")
plt.show()
