import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import os
import sys

# Add the parent directory to the path to import Visualizer
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Import the RLLogger from Visualizer
from Visualizer.logger import RLLogger  # noqa: E402

seed = 0
ENV_NAME = "MountainCar-v0"
LOG_DIR = os.path.join(current_dir, "logs")
POLICY_NAME = "DQN"
MODEL_DIR = os.path.join(current_dir, "models")
RENDER_MODE = True

if RENDER_MODE:
    env = gym.make(ENV_NAME, render_mode="human")
else:
    env = gym.make(ENV_NAME)

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


######################################################################
# Replay Memory
# -------------------

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# DQN algorithm
# -------------


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


######################################################################
# Training
# --------

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0  # start with full exploration
EPS_END = 0.01  # minimal exploration level
EPS_DECAY = 20000  # slower decay for prolonged exploration
TAU = 0.005
LR = 1e-3  # slightly higher learning rate for MountainCar

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

try:
    policy_net.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, f"{POLICY_NAME}_{ENV_NAME}.pth"))
    )
    target_net.load_state_dict(policy_net.state_dict())
    print("Model loaded successfully")
except FileNotFoundError:
    print("No model found, training from scratch")
    target_net.load_state_dict(policy_net.state_dict())


optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


episode_rewards = []


def plot_durations(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


######################################################################
# Training loop
# -------------
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 10000)
    optimizer.step()

    return loss.item()


######################################################################
# Main training loop
# -------------------

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 500
else:
    num_episodes = 500

# Create output directory for logs

os.makedirs(LOG_DIR, exist_ok=True)
# Initialize logger
logger = RLLogger(
    policy=POLICY_NAME, environment=ENV_NAME, seed=seed, output_dir=LOG_DIR
)

# Set random seeds for reproducibility
random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)

total_timesteps = 0
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    episode_reward = 0
    episode_loss = 0
    episode_timesteps = 0

    for t in count():
        action = select_action(state)
        observation, base_reward, terminated, truncated, _ = env.step(action.item())
        # Reward shaping: add bonus proportional to horizontal position
        position = observation[0]
        alpha = 1.0  # increased shaping bonus
        shaped_reward = base_reward + alpha * (position + 0.5)
        reward = torch.tensor([shaped_reward], device=device)
        done = terminated or truncated

        episode_reward += shaped_reward
        episode_timesteps += 1
        total_timesteps += 1

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model()
        if loss is not None:
            episode_loss += loss

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_rewards.append(episode_reward)
            plot_durations()

            # Log episode data
            logger.log_episode(
                total_timesteps=total_timesteps,
                episode_num=i_episode,
                episode_timesteps=episode_timesteps,
                reward=episode_reward,
            )

            # Save log file every 25 episodes and at the end of training
            if i_episode % 25 == 0 or i_episode == num_episodes - 1:
                log_path = logger.save()
                print(f"Saved log to {log_path}")

            break

print("Complete")
plot_durations(show_result=True)

# Save the final log file
final_log_path = logger.save()
print(f"Training complete! Final log saved to: {final_log_path}")

torch.save(
    policy_net.state_dict(), os.path.join(MODEL_DIR, f"{POLICY_NAME}_{ENV_NAME}.pth")
)
print(f"Saved policy_net weights to '{POLICY_NAME}_{ENV_NAME}.pth'")

plt.ioff()
plt.show()
