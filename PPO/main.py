import numpy as np
import torch
import gym
import argparse
import os
import time
import sys

# Add the parent directory to path to import from visualizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Visualizer.logger import RLLogger

# Import PPO variants
from ppo_base import PPO_Base
from ppo_clip import PPO_Clip
from ppo_kl import PPO_KL
from buffer import PPOBuffer

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), deterministic=True)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="PPO-Clip")                 # Policy name (PPO-Base, PPO-Clip, PPO-KL)
    parser.add_argument("--env", default="MountainCarContinuous-v0")    # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--steps_per_epoch", default=4000, type=int)    # Steps per training epoch
    parser.add_argument("--epochs", default=250, type=int)              # Training epochs
    parser.add_argument("--eval_freq", default=10000, type=int)         # How often (steps) to evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)       # Max time steps to run environment
    parser.add_argument("--gamma", default=0.99, type=float)            # Discount factor
    parser.add_argument("--gae_lambda", default=0.95, type=float)       # GAE parameter for advantage estimation
    parser.add_argument("--clip_param", default=0.2, type=float)        # Clipping parameter epsilon for PPO-Clip
    parser.add_argument("--kl_target", default=0.01, type=float)        # Target KL divergence for PPO-KL
    parser.add_argument("--mini_batch_size", default=64, type=int)      # Mini batch size for optimizer
    parser.add_argument("--train_epochs", default=10, type=int)         # Number of epochs to optimize for each update
    parser.add_argument("--lr", default=3e-4, type=float)               # Learning rate for optimizer
    parser.add_argument("--value_coef", default=0.5, type=float)        # Value function coefficient
    parser.add_argument("--entropy_coef", default=0.01, type=float)     # Entropy coefficient
    parser.add_argument("--save_model", action="store_true")            # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                     # Model load file name, "" doesn't load
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    # Create output directories if they don't exist
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # Initialize environment
    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get environment dimensions and limits
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Get max episode steps for the environment
    max_ep_len = env._max_episode_steps if hasattr(env, '_max_episode_steps') else 1000
    
    # Initialize logger
    logger = RLLogger(args.policy, args.env, args.seed)

    # Initialize policy based on chosen algorithm
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.gamma,
        "lr": args.lr,
        "gae_lambda": args.gae_lambda,
        "epochs": args.train_epochs,
        "mini_batch_size": args.mini_batch_size,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
    }

    if args.policy == "PPO-Clip":
        kwargs["clip_param"] = args.clip_param
        policy = PPO_Clip(**kwargs)
    elif args.policy == "PPO-KL":
        kwargs["kl_target"] = args.kl_target
        policy = PPO_KL(**kwargs)
    elif args.policy == "PPO-Base":
        policy = PPO_Base(**kwargs)
    else:
        raise ValueError(f"Unsupported policy: {args.policy}")

    # Load pretrained model if specified
    if args.load_model != "":
        policy.load(f"./models/{args.load_model}")

    # Setup episode recording variables
    timesteps_so_far = 0
    episode_num = 0
    num_updates = 0
    
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]
    
    # Initialize buffer
    buffer_size = args.steps_per_epoch
    buffer = PPOBuffer(state_dim, action_dim, buffer_size, args.gamma, args.gae_lambda)

    # Run training loop
    for epoch in range(args.epochs):
        # Reset buffer
        buffer_ptr = 0
        
        # Collect data for one epoch
        state, ep_ret, ep_len = env.reset(), 0, 0
        
        for t in range(args.steps_per_epoch):
            timesteps_so_far += 1
            
            # Select action
            action = policy.select_action(np.array(state))
            
            # Get value estimate from critic
            value = policy.evaluate(np.array(state))
            
            # Convert action to torch tensor for getting log probability
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(policy.actor.device)
            action_tensor, log_prob = policy.actor.get_action(state_tensor)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            ep_ret += reward
            ep_len += 1
            
            # Store in buffer
            buffer.store(
                state, 
                action, 
                reward, 
                value, 
                log_prob.item(), 
                done, 
                next_state
            )
            buffer_ptr += 1
            
            # Update state
            state = next_state
            
            # End of trajectory handling
            timeout = ep_len == max_ep_len
            terminal = done or timeout
            
            if terminal or buffer_ptr >= buffer_size:
                # If trajectory didn't end, bootstrap value target
                last_value = 0
                if not done:
                    last_value = policy.evaluate(np.array(state))
                
                buffer.finish_path(last_value=last_value)
                
                if terminal:
                    # Log episode stats
                    logger.log_episode(timesteps_so_far, episode_num, ep_len, ep_ret)
                    
                    # Reset environment
                    state, ep_ret, ep_len = env.reset(), 0, 0
                    episode_num += 1
        
        # Update policy
        num_updates += 1
        metrics = policy.train(buffer)
        
        # Log training metrics
        print(f"Epoch: {epoch+1}/{args.epochs}")
        print(f"Policy Loss: {metrics['policy_loss']:.3f}, Value Loss: {metrics['value_loss']:.3f}")
        print(f"KL Divergence: {metrics['kl_div']:.5f}, Entropy: {metrics['entropy']:.3f}")
        if 'clip_fraction' in metrics:
            print(f"Clip Fraction: {metrics['clip_fraction']:.3f}")
        if 'beta' in metrics:
            print(f"KL Penalty Beta: {metrics['beta']:.5f}")
        print("---------------------------------------")
        
        # Evaluate episode
        if timesteps_so_far % args.eval_freq < args.steps_per_epoch:
            avg_reward = eval_policy(policy, args.env, args.seed)
            evaluations.append(avg_reward)
            logger.log_evaluation(timesteps_so_far, 10, avg_reward)
            
            # Save model and logs
            if args.save_model:
                policy.save(f"./models/{file_name}")
            
            logger.save()
    
    # Save final model and logs
    if args.save_model:
        policy.save(f"./models/{file_name}_final")
        
    logger.save()