import os
import gym
import numpy as np
import torch
import sys

# Add parent directory to path to use custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Visualizer.logger import RLLogger

# PPO2 network and policy implementation
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Constants
HIDDEN_SIZE = 64
EPS = 1e-8  # Small constant to avoid division by zero

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Policy, self).__init__()
        self.max_action = max_action
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh()
        )
        # Mean and log_std for continuous actions
        self.mean = nn.Linear(HIDDEN_SIZE, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1)
        )
    
    def forward(self, x):
        # Get mean and std for the action distribution
        a = self.actor(x)
        mean = self.mean(a)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        
        # Get value function prediction
        v = self.critic(x)
        return mean, std, v
    
    def get_action(self, state, deterministic=False):
        mean, std, _ = self(state)
        
        if deterministic:
            action = mean
        else:
            # Sample from Normal distribution
            normal = Normal(mean, std)
            action = normal.sample()
            
        return action * self.max_action, normal.log_prob(action).sum(dim=-1)
    
    def evaluate(self, state):
        _, _, v = self(state)
        return v.squeeze(-1)

class PPO:
    def __init__(self, state_dim, action_dim, max_action, clip_param=0.2, lr=3e-4, gamma=0.99, gae_lambda=0.95):
        self.policy = Policy(state_dim, action_dim, max_action)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.clip_param = clip_param
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action, _ = self.policy.get_action(state, deterministic)
        return action.cpu().numpy().flatten()
    
    def train(self, replay_buffer):
        # Get states, actions, etc. from buffer
        states, actions, rewards, dones, next_states, log_probs, values = replay_buffer.get()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        
        # Compute advantages using GAE
        advantages = torch.FloatTensor(replay_buffer.advantages).to(self.device)
        returns = torch.FloatTensor(replay_buffer.returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
        
        # PPO update
        mean, std, current_values = self.policy(states)
        dist = Normal(mean, std)
        current_log_probs = dist.log_prob(actions / self.policy.max_action).sum(-1)
        
        # Compute ratio
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Compute surrogate losses
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        
        # Policy loss
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(current_values, returns)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # KL divergence
            kl = 0.5 * ((mean - mean).pow(2) + std.pow(2) / std.pow(2) - 1 + std.log() - std.log()).sum(-1).mean()
            # Clip fraction
            clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.clip_param).float())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl': kl.item(),
            'clip_frac': clip_frac.item()
        }
    
    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)
        
    def load(self, filepath):
        self.policy.load_state_dict(torch.load(filepath))

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size, gamma, gae_lambda):
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        
    def store(self, state, action, reward, done, next_state, log_prob, value):
        idx = self.ptr % self.buffer_size
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.next_states[idx] = next_state
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_advantages(self, last_value):
        path_slice = slice(self.ptr - self.size, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value * (1 - self.dones[self.ptr - 1]))
        values = np.append(self.values[path_slice], last_value)
        
        # GAE calculation
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - self.dones[path_slice]) - values[:-1]
        
        advantages = np.zeros_like(deltas)
        lastgaelam = 0
        for t in reversed(range(len(deltas))):
            lastgaelam = deltas[t] + self.gamma * self.gae_lambda * (1 - self.dones[path_slice][t]) * lastgaelam
            advantages[t] = lastgaelam
        
        self.advantages = advantages
        self.returns = advantages + self.values[path_slice]
        
    def get(self):
        path_slice = slice(self.ptr - self.size, self.ptr)
        return (
            self.states[path_slice],
            self.actions[path_slice],
            self.rewards[path_slice],
            self.dones[path_slice],
            self.next_states[path_slice],
            self.log_probs[path_slice],
            self.values[path_slice]
        )
    
    def clear(self):
        self.ptr = 0
        self.size = 0

def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(state, deterministic=True)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    
    avg_reward /= eval_episodes
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def train_ppo(env_name, seed=0, total_timesteps=1000000):
    # Create environment
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # PPO hyperparameters
    steps_per_epoch = 4000
    batch_size = 64
    n_epochs = 10
    clip_param = 0.2
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    
    # Initialize policy
    policy = PPO(state_dim, action_dim, max_action, 
                clip_param=clip_param, lr=lr, gamma=gamma, gae_lambda=gae_lambda)
    
    # Initialize buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, steps_per_epoch, gamma, gae_lambda)
    
    # Initialize logger
    logger = RLLogger('PPO2', env_name, seed)
    
    # Training loop
    timesteps_so_far = 0
    episode_num = 0
    
    # Setup output directories
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # Initial evaluation
    evaluations = [eval_policy(policy, env_name, seed)]
    logger.log_evaluation(0, 10, evaluations[-1])
    
    state, done = env.reset(), False
    episode_reward = 0
    episode_len = 0
    
    for t in range(total_timesteps):
        timesteps_so_far += 1
        episode_len += 1
        
        # Select action
        action = policy.select_action(state)
        
        # Get value estimate
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(policy.device)
        value = policy.policy.evaluate(state_tensor).cpu().item()
        
        # Convert action to torch tensor for getting log probability
        action_tensor, log_prob = policy.policy.get_action(state_tensor)
        
        # Execute action
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        
        # Store in buffer
        replay_buffer.store(
            state,
            action,
            reward,
            done,
            next_state,
            log_prob.cpu().item(),
            value
        )
        
        # Update state
        state = next_state
        
        # End of episode handling
        timeout = episode_len == env._max_episode_steps
        terminal = done or timeout
        
        if terminal or (t > 0 and t % steps_per_epoch == 0):
            # If trajectory didn't end, bootstrap value target
            if not terminal:
                state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(policy.device)
                last_value = policy.policy.evaluate(state_tensor).cpu().item()
            else:
                last_value = 0
            
            # Compute advantages and returns
            replay_buffer.compute_advantages(last_value)
            
            # Update policy
            metrics = policy.train(replay_buffer)
            
            if terminal:
                # Log episode stats
                logger.log_episode(timesteps_so_far, episode_num, episode_len, episode_reward)
                
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_len = 0
                episode_num += 1
            
            # Clear buffer
            replay_buffer.clear()
            
            # Print training info
            if t % (5 * steps_per_epoch) == 0:
                print(f"Timestep: {t}/{total_timesteps}")
                print(f"Policy Loss: {metrics['policy_loss']:.4f}, Value Loss: {metrics['value_loss']:.4f}")
                print(f"KL Divergence: {metrics['kl']:.4f}, Clip Fraction: {metrics['clip_frac']:.4f}")
                
                # Evaluate policy
                eval_reward = eval_policy(policy, env_name, seed)
                evaluations.append(eval_reward)
                logger.log_evaluation(timesteps_so_far, 10, eval_reward)
                
                # Save policy and logs
                policy.save(f"./models/ppo2_{env_name}_{seed}")
                logger.save()
    
    # Final evaluation
    eval_reward = eval_policy(policy, env_name, seed)
    evaluations.append(eval_reward)
    logger.log_evaluation(timesteps_so_far, 10, eval_reward)
    
    # Save final policy and logs
    policy.save(f"./models/ppo2_{env_name}_{seed}_final")
    logger.save()
    
    return policy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MountainCarContinuous-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1000000)
    args = parser.parse_args()
    
    train_ppo(args.env, seed=args.seed, total_timesteps=args.timesteps)