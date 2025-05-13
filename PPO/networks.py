import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization for the policy and value networks
    As mentioned in the PPO paper appendix
    """
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    """
    Diagonal Gaussian policy network for continuous action spaces as described in the PPO paper
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.fc1 = init_weights(nn.Linear(state_dim, hidden_dim))
        self.fc2 = init_weights(nn.Linear(hidden_dim, hidden_dim))
        self.mean = init_weights(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        self.max_action = max_action

    def forward(self, state):
        """Forward pass to get mean and std of the policy distribution"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        
        # Ensure log_std stays within a reasonable range as mentioned in the PPO paper
        log_std = self.log_std.expand_as(mean).clamp(-20, 2)
        std = torch.exp(log_std)
        
        return mean, std
        
    def get_action(self, state, deterministic=False):
        """
        Sample action from the policy distribution
        
        Args:
            state: Current state
            deterministic: If True, return the mean action
            
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
        """
        mean, std = self.forward(state)
        distribution = Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            action = distribution.sample()
        
        log_prob = distribution.log_prob(action).sum(axis=-1)
        action = self.max_action * torch.tanh(action)  # Scale to action space
        
        return action, log_prob
    
    def log_prob(self, state, action):
        """
        Calculate log probability of an action given a state
        Used for importance sampling in PPO
        
        Args:
            state: State
            action: Action
            
        Returns:
            log_prob: Log probability of the action
        """
        mean, std = self.forward(state)
        
        # Normalize action based on tanh bounds
        action = action / self.max_action
        action = torch.atanh(torch.clamp(action, -0.999, 0.999))
        
        distribution = Normal(mean, std)
        log_prob = distribution.log_prob(action).sum(axis=-1)
        
        return log_prob

class Critic(nn.Module):
    """Value function approximator"""
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.fc1 = init_weights(nn.Linear(state_dim, hidden_dim))
        self.fc2 = init_weights(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = init_weights(nn.Linear(hidden_dim, 1), std=1.0)
        
    def forward(self, state):
        """Forward pass to get state value"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.fc3(x).squeeze(-1)
        return v