import json
import os
import time
from datetime import datetime


class RLLogger:
    def __init__(self, policy, environment, seed, output_dir="./logs"):
        """
        Args:
            policy (str): The name of the policy being used
            environment (str): The name of the environment
            seed (int): Random seed used for the experiment
            output_dir (str): Directory to save log files
        """
        self.data = {
            "experiment": {
                "policy": policy,
                "environment": environment,
                "seed": seed,
                "start_time": datetime.now().isoformat(),
            },
            "episodes": [],
            "evaluations": [],
        }

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.filename = f"{policy}_{environment}_seed-{seed}_{int(time.time())}"

    def log_episode(self, total_timesteps, episode_num, episode_timesteps, reward):
        """
        Log data for a completed episode

        Args:
            total_timesteps (int): Total timesteps so far in the experiment
            episode_num (int): Current episode number
            episode_timesteps (int): Timesteps in this episode
            reward (float): Total reward for this episode
        """
        episode_data = {
            "total_timesteps": total_timesteps,
            "episode_num": episode_num,
            "episode_timesteps": episode_timesteps,
            "reward": float(reward),
        }

        self.data["episodes"].append(episode_data)

        # print(
        #     f"Total T: {total_timesteps} Episode Num: {episode_num} "
        #     f"Episode T: {episode_timesteps} Reward: {reward:.3f}"
        # )

    def log_evaluation(self, at_timesteps, eval_episodes, avg_reward):
        """
        Log data from policy evaluation

        Args:
            at_timesteps (int): Timestep at which the evaluation was performed
            eval_episodes (int): Number of episodes used for evaluation
            avg_reward (float): Average reward across evaluation episodes
        """
        eval_data = {
            "at_timesteps": at_timesteps,
            "evaluation_over_" + str(eval_episodes) + "_episodes": float(avg_reward),
        }

        self.data["evaluations"].append(eval_data)

        # print("---------------------------------------")
        # print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        # print("---------------------------------------")

    def save(self, filename=None):
        """
        Save the logged data as a JSON file

        Args:
            filename (str, optional): Filename to save as. If None, uses the default name
                                     based on policy, environment and seed.

        Returns:
            str: Path to the saved file
        """
        if filename is None:
            filename = self.filename

        if not filename.endswith(".json"):
            filename += ".json"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(self.data, f, indent=2)

        return filepath

    def get_data(self):
        """
        Get the current data

        Returns:
            dict: The structured data collected so far
        """
        return self.data
