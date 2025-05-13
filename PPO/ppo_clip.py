import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from networks import Actor, Critic
from buffer import PPOBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO with Clipping implementation (PPO-Clip)
# Based on Section 4 of the paper by Schulman et al. (2017)
# Paper: https://arxiv.org/abs/1707.06347
class PPO_Clip(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        lr=3e-4,
        gae_lambda=0.95,
        epochs=10,
        mini_batch_size=64,
        value_coef=0.5,
        entropy_coef=0.01,
        clip_param=0.2,
    ):
        """
        Initialize PPO with clipped surrogate objective
        
        Args:
            state_dim (int): Dimension of state
            action_dim (int): Dimension of action
            max_action (float): Maximum action value
            discount (float): Discount factor gamma
            lr (float): Learning rate for optimizer
            gae_lambda (float): Lambda for GAE
            epochs (int): Number of epochs to optimize on each update
            mini_batch_size (int): Mini-batch size for updates
            value_coef (float): Value loss coefficient (c1)
            entropy_coef (float): Entropy coefficient (c2)
            clip_param (float): Clipping parameter epsilon for the surrogate objective
        """
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.clip_param = clip_param
        
        # Track metrics
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy = 0
        self.kl_div = 0
        self.clip_fraction = 0 # Fraction of time clipping is activated

    def select_action(self, state, deterministic=False):
        """
        Select an action from the policy
        
        Args:
            state: Current state
            deterministic: If True, return deterministic action
            
        Returns:
            action: Selected action
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, _ = self.actor.get_action(state, deterministic)
        return action.cpu().data.numpy().flatten()

    def evaluate(self, state):
        """
        Get state value from critic
        
        Args:
            state: Current state
            
        Returns:
            value: State value estimate
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.critic(state).cpu().data.numpy().flatten()[0]
        
    def train(self, buffer):
        """
        Train the PPO agent using collected data
        
        Args:
            buffer: PPOBuffer containing collected trajectories
            
        Returns:
            dict: Dictionary of metrics from training
        """
        # Get data from buffer
        data = buffer.get()
        states = data['states']
        actions = data['actions']
        logprobs_old = data['logprobs']
        returns = data['returns']
        advantages = data['advantages']
        
        # Track old policy for KL divergence calculation
        with torch.no_grad():
            old_mean, old_std = self.actor(states)
        
        # Track metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []
        clip_fractions = []
        
        # Train for specified number of epochs
        for epoch in range(self.epochs):
            # Generate random permutation of indices for mini-batches
            indices = torch.randperm(states.shape[0])
            
            # Train on mini-batches
            for start in range(0, states.shape[0], self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                state_batch = states[batch_indices]
                action_batch = actions[batch_indices]
                logprob_old_batch = logprobs_old[batch_indices]
                advantage_batch = advantages[batch_indices]
                return_batch = returns[batch_indices]
                
                # Calculate log probabilities and entropy of current policy
                logprob_batch = self.actor.log_prob(state_batch, action_batch)
                
                # Calculate probability ratio r(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(logprob_batch - logprob_old_batch)
                
                # Clipped surrogate objective as described in PPO paper Section 4:
                # L^CLIP(θ) = Ê_t[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage_batch
                
                # Calculate the clipped policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate clipping fraction (diagnostic metric)
                clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                
                # Calculate KL divergence between old and new policy
                mean, std = self.actor(state_batch)
                kl_div = torch.log(std / old_std[batch_indices]) + \
                         (old_std[batch_indices].pow(2) + (old_mean[batch_indices] - mean).pow(2)) / \
                         (2.0 * std.pow(2)) - 0.5
                kl_div = kl_div.sum(1).mean()
                
                # Calculate entropy
                entropy = -logprob_batch.mean()
                
                # Calculate value loss
                value_batch = self.critic(state_batch)
                value_loss = F.mse_loss(value_batch, return_batch)
                
                # Calculate total loss: L^CLIP+VF+S(θ) from PPO paper
                # L^CLIP+VF+S(θ) = Ê_t[L^CLIP(θ) - c1 * (V_θ(s_t) - V_target)² + c2 * S[π_θ](s_t)]
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update actor and critic networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
                kl_divs.append(kl_div.item())
                clip_fractions.append(clip_fraction)
        
        # Store average metrics
        self.policy_loss = np.mean(policy_losses)
        self.value_loss = np.mean(value_losses)
        self.entropy = np.mean(entropy_losses)
        self.kl_div = np.mean(kl_divs)
        self.clip_fraction = np.mean(clip_fractions)
        
        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "kl_div": self.kl_div,
            "clip_fraction": self.clip_fraction
        }
            
    def save(self, filename):
        """Save model parameters"""
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        """Load model parameters"""
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))