import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Define the transition tuple 

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Define the structure of the replay memory

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
    
    # Define the Deep Q-Network using PyTorch nn.Module
    class DQN(nn.Module):

        def __init__(self, n_observations, n_actions):
            super(DQN, self).__init__()
            self.layer1 = nn.Linear(n_observations, 128) # Input observations (4) to hidden layer (128)
            self.layer2 = nn.Linear(128, 128) # Hidden layer (128) to hidden layer (128)
            self.layer3 = nn.Linear(128, n_actions) # Output hidden layer (128) to output actions (2)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer1(x))
            return self.layer3(x)
        

# Training
# Hyperparameters

BATCH_SIZE = 128 # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99 # GAMMA is the discount factor as mentioned in the previous section
EPS_START = 0.9 # EPS_START is the starting value of epsilon
EPS_END = 0.05 # EPS_END is the final value of epsilon
EPS_DECAY = 1000 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # TAU is the update rate of the target network
LR = 1e-4 # LR is the learning rate of the ``AdamW`` optimizer

# Get the number of actions from the gym action space
n_actions = env.action_space.n
# Get the number of state observations from the gym observation space
state,info = env.reset() # Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information.
n_observations = len(state) # Number of observations is the length of the state

policy_net = DQN(n_observations, n_actions).to(device) # Create the policy network
target_net = DQN(n_observations, n_actions).to(device) # Create the target network
# Load saved weights if they exist
try:
    policy_net.load_state_dict(torch.load('dqn_cartpole.pth'))
    target_net.load_state_dict(policy_net.state_dict())  # Sync target network
    print("Loaded pretrained weights!")
except FileNotFoundError:
    print("No saved weights found. Training from scratch.")
    target_net.load_state_dict(policy_net.state_dict())  # Default sync

optimizer = optim.AdamW(policy_net.parameters(), lr=LR) # Create the optimizer
memory = ReplayMemory(10000) # Create the replay memory

steps_done = 0 # Used to track the steps done during training (decay epsilon to transition from exploration to exploitation)

def select_action(state):
    global steps_done
    sample = random.random() # Sample a random number between 0 and 1
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY) # Calculate the epsilon threshold
    steps_done += 1 # Increment the steps done
    if sample > eps_threshold: # The odds that the sample is greater than the epsilon is increasingly likely as training progresses
        with torch.no_grad(): # No need to track gradients for this operation, would slow down training
            return policy_net(state).max(1).indices.view(1,1) # Return the action with the highest Q-value
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long) # Return a random action sampled from the action space  
    
episode_durations = []

# For plot visualization
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Training loop
def optimize_model():
    if len(memory) < BATCH_SIZE: # Check if there are enough experiences in memory
        return
    transitions = memory.sample(BATCH_SIZE) # Sample a batch of experiences from memory
    batch = Transition(*zip(*transitions)) # Unzip the batch into a list of transitions

    # Compute a mask to check if the next state is a final state
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                            device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch) # Get the Q-values for the actions actually taken

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final. This is the addition of the 2015
    # paper, "Human-level control through deep reinforcement learning" by Mnih et al.
    # https://arxiv.org/abs/1509.06461
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# Main training loop
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
# Save the policy network's state dictionary
torch.save(policy_net.state_dict(), 'dqn_cartpole.pth')
print("Saved policy_net weights to 'dqn_cartpole.pth'")

plot_durations(show_result=True)
plt.ioff()
plt.show()