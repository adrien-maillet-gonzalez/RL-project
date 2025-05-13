import numpy as np
import torch
import scipy.signal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOBuffer:
    """
    Buffer for storing trajectories and computing Generalized Advantage Estimation (GAE)
    """
    def __init__(self, state_dim, action_dim, buffer_size, gamma=0.99, gae_lambda=0.95):
        """
        Initialize PPO buffer for collecting trajectories and computing advantages
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            buffer_size (int): Maximum buffer size for storing trajectories in one update
            gamma (float): Discount factor for rewards (γ)
            gae_lambda (float): Lambda parameter for GAE (λ)
        """
        self.state_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.next_state_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
        self.value_buf = np.zeros(buffer_size, dtype=np.float32)
        self.return_buf = np.zeros(buffer_size, dtype=np.float32)
        self.advantage_buf = np.zeros(buffer_size, dtype=np.float32)
        self.logprob_buf = np.zeros(buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(buffer_size, dtype=np.float32)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ptr, self.path_start_idx, self.max_size = 0, 0, buffer_size
        self.device = device

    def store(self, state, action, reward, value, logprob, done, next_state):
        """
        Store a transition in the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate from the critic
            logprob: Log probability of the action
            done: Boolean indicating if the episode is done
            next_state: Next state
        """
        assert self.ptr < self.max_size  # Buffer has to have room
        
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.value_buf[self.ptr] = value
        self.logprob_buf[self.ptr] = logprob
        self.done_buf[self.ptr] = done
        self.next_state_buf[self.ptr] = next_state
        
        self.ptr += 1

    def finish_path(self, last_value=0):
        """
        Call this at the end of a trajectory to compute advantages and returns
        
        Args:
            last_value (float): Value estimate for the last (incomplete) state
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.reward_buf[path_slice], last_value)
        values = np.append(self.value_buf[path_slice], last_value)
        dones = np.append(self.done_buf[path_slice], 0)
        
        # GAE calculation for this path
        deltas = rewards[:-1] + self.gamma * (1 - dones[:-1]) * values[1:] - values[:-1]
        
        self.advantage_buf[path_slice] = self._discount_cumsum(
            deltas, self.gamma * self.gae_lambda
        )
        
        # Compute returns for TD(λ) targets
        self.return_buf[path_slice] = self._discount_cumsum(
            rewards, self.gamma
        )[:-1]
        
        self.path_start_idx = self.ptr

    def get(self, normalize_adv=True):
        """
        Get all data from the buffer and normalize advantages
        
        Args:
            normalize_adv (bool): Whether to normalize advantages
            
        Returns:
            dict: Dictionary containing buffer data as torch tensors on device
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize advantage if requested
        if normalize_adv:
            adv_mean = np.mean(self.advantage_buf)
            adv_std = np.std(self.advantage_buf)
            self.advantage_buf = (self.advantage_buf - adv_mean) / (adv_std + 1e-8)
        
        data = dict(
            states=torch.tensor(self.state_buf, dtype=torch.float32).to(self.device),
            actions=torch.tensor(self.action_buf, dtype=torch.float32).to(self.device),
            logprobs=torch.tensor(self.logprob_buf, dtype=torch.float32).to(self.device),
            returns=torch.tensor(self.return_buf, dtype=torch.float32).to(self.device),
            advantages=torch.tensor(self.advantage_buf, dtype=torch.float32).to(self.device),
            values=torch.tensor(self.value_buf, dtype=torch.float32).to(self.device),
        )
        
        return data
    
    def _discount_cumsum(self, x, discount):
        """
        Compute discounted cumulative sum of a time series
        
        Args:
            x (np.array): Input time series
            discount (float): Discount factor
            
        Returns:
            np.array: Discounted cumulative sum
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]