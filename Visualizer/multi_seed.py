import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
from collections import defaultdict

from .visualizer import load_data


def align_episode_data(all_data):
    """
    Args:
        all_data (list): List of experiment data dictionaries

    Returns:
        dict: Dictionary with episode numbers as keys and lists of rewards as values
    """
    episode_rewards = defaultdict(list)

    for data in all_data:
        for episode in data["episodes"]:
            episode_num = episode["episode_num"]
            reward = episode["reward"]
            episode_rewards[episode_num].append(reward)

    return episode_rewards


def align_evaluation_data(all_data):
    """
    Args:
        all_data (list): List of experiment data dictionaries

    Returns:
        dict: Dictionary with timesteps as keys and lists of evaluation scores as values
    """
    eval_scores = defaultdict(list)

    for data in all_data:
        for eval_data in data["evaluations"]:
            timestep = eval_data["at_timesteps"]
            eval_key = [k for k in eval_data.keys() if k.startswith("evaluation_over")][
                0
            ]
            score = eval_data[eval_key]
            eval_scores[timestep].append(score)

    return eval_scores


def plot_individual_seeds(all_data, save_path=None, show=True):
    """
    Args:
        all_data (list): List of experiment data dictionaries
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
    """
    plt.figure(figsize=(12, 6))

    colors = ["b", "g", "r", "c", "m", "y", "k"]

    policy = all_data[0]["experiment"]["policy"]
    environment = all_data[0]["experiment"]["environment"]

    for i, data in enumerate(all_data):
        seed = data["experiment"]["seed"]
        color = colors[i % len(colors)]

        episode_nums = [e["episode_num"] for e in data["episodes"]]
        rewards = [e["reward"] for e in data["episodes"]]

        plt.plot(episode_nums, rewards, color=color, alpha=0.5, label=f"Seed {seed}")

    plt.title(f"Episode Rewards by Seed - {policy} on {environment}")
    plt.xlabel("Episode Number")
    plt.ylabel("Total Reward")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_mean_episode_rewards(all_data, save_path=None, show=True):
    """
    Args:
        all_data (list): List of experiment data dictionaries
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
    """
    episode_rewards = align_episode_data(all_data)

    episode_nums = sorted(episode_rewards.keys())
    mean_rewards = []
    std_rewards = []

    for ep_num in episode_nums:
        rewards = episode_rewards[ep_num]
        mean_rewards.append(np.mean(rewards))
        if len(rewards) > 1:
            std_rewards.append(np.std(rewards))
        else:
            std_rewards.append(0)

    policy = all_data[0]["experiment"]["policy"]
    environment = all_data[0]["experiment"]["environment"]

    plt.figure(figsize=(12, 6))
    plt.plot(episode_nums, mean_rewards, "b-", label="Mean Reward")

    if len(all_data) > 1:
        plt.fill_between(
            episode_nums,
            np.array(mean_rewards) - np.array(std_rewards),
            np.array(mean_rewards) + np.array(std_rewards),
            alpha=0.2,
            color="b",
            label="±1 Std Dev",
        )

    if len(episode_nums) > 1:
        z = np.polyfit(episode_nums, mean_rewards, 1)
        p = np.poly1d(z)
        plt.plot(
            episode_nums,
            p(episode_nums),
            "r--",
            alpha=0.8,
            label=f"Trend: {z[0]:.2f}x + {z[1]:.2f}",
        )

    plt.title(f"Mean Episode Rewards Across Seeds - {policy} on {environment}")
    plt.xlabel("Episode Number")
    plt.ylabel("Mean Total Reward")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_mean_evaluation_performance(all_data, save_path=None, show=True):
    """
    Args:
        all_data (list): List of experiment data dictionaries
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
    """
    eval_scores = align_evaluation_data(all_data)

    timesteps = sorted(eval_scores.keys())
    mean_scores = []
    std_scores = []

    for t in timesteps:
        scores = eval_scores[t]
        mean_scores.append(np.mean(scores))
        if len(scores) > 1:
            std_scores.append(np.std(scores))
        else:
            std_scores.append(0)

    policy = all_data[0]["experiment"]["policy"]
    environment = all_data[0]["experiment"]["environment"]

    eval_key = [
        k
        for k in all_data[0]["evaluations"][0].keys()
        if k.startswith("evaluation_over")
    ][0]
    eval_episodes = eval_key.split("_")[2]

    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, mean_scores, "bo-", label="Mean Evaluation Score")

    if len(all_data) > 1:
        plt.fill_between(
            timesteps,
            np.array(mean_scores) - np.array(std_scores),
            np.array(mean_scores) + np.array(std_scores),
            alpha=0.2,
            color="b",
            label="±1 Std Dev",
        )

    plt.title(f"Mean Evaluation Performance Across Seeds - {policy} on {environment}")
    plt.xlabel("Timesteps")
    plt.ylabel(f"Mean Evaluation Score (over {eval_episodes} episodes)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_mean_rolling_rewards(all_data, window=10, save_path=None, show=True):
    """
    Args:
        all_data (list): List of experiment data dictionaries
        window (int): Size of the rolling window
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
    """
    episode_rewards = align_episode_data(all_data)

    episode_nums = sorted(episode_rewards.keys())
    mean_rewards = []

    for ep_num in episode_nums:
        rewards = episode_rewards[ep_num]
        mean_rewards.append(np.mean(rewards))

    rolling_rewards = []
    for i in range(len(mean_rewards)):
        if i < window - 1:
            rolling_rewards.append(np.mean(mean_rewards[: i + 1]))
        else:
            rolling_rewards.append(np.mean(mean_rewards[i - window + 1 : i + 1]))

    policy = all_data[0]["experiment"]["policy"]
    environment = all_data[0]["experiment"]["environment"]

    plt.figure(figsize=(12, 6))
    plt.plot(episode_nums, mean_rewards, "b-", alpha=0.3, label="Mean Episode Rewards")
    plt.plot(
        episode_nums, rolling_rewards, "r-", label=f"{window}-Episode Rolling Average"
    )

    plt.title(f"Mean Episode Rewards Across Seeds - {policy} on {environment}")
    plt.xlabel("Episode Number")
    plt.ylabel("Mean Reward")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def visualize_multi_seed(
    json_files=None, json_pattern=None, output_dir="./plots", rolling_window=10
):
    """
    Args:
        json_files (list, optional): List of JSON file paths
        json_pattern (str, optional): Glob pattern to match JSON files
        output_dir (str): Directory to save plots
        rolling_window (int): Window size for rolling average
    """
    if json_files is None and json_pattern is None:
        raise ValueError("Either json_files or json_pattern must be provided")

    if json_files is None:
        json_files = glob.glob(json_pattern)

    if not json_files:
        print("No JSON files found. Please check your input.")
        return

    print(f"Processing {len(json_files)} JSON files:")
    for f in json_files:
        print(f"  - {f}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_data = [load_data(f) for f in json_files]

    policy = all_data[0]["experiment"]["policy"]
    environment = all_data[0]["experiment"]["environment"]
    base_name = f"{policy}_{environment}_mean"

    plot_individual_seeds(
        all_data,
        save_path=os.path.join(output_dir, f"{base_name}_individual_seeds.png"),
    )

    plot_mean_episode_rewards(
        all_data, save_path=os.path.join(output_dir, f"{base_name}_rewards.png")
    )

    plot_mean_evaluation_performance(
        all_data, save_path=os.path.join(output_dir, f"{base_name}_eval.png")
    )

    plot_mean_rolling_rewards(
        all_data,
        window=rolling_window,
        save_path=os.path.join(output_dir, f"{base_name}_rolling_{rolling_window}.png"),
    )

    print(f"Plots saved to {output_dir}")


def main():
    """Command-line interface for visualizing multi-seed results"""
    parser = argparse.ArgumentParser(
        description="Visualize mean results across multiple seeds"
    )
    parser.add_argument(
        "--json_files", nargs="+", help="Paths to the JSON results files"
    )
    parser.add_argument(
        "--json_pattern",
        help="Glob pattern to match JSON files (e.g., 'logs/TD3_Pendulum-v1_seed-*.json')",
    )
    parser.add_argument(
        "--output_dir", "-o", default="./plots", help="Directory to save plots"
    )
    parser.add_argument(
        "--rolling_window",
        "-w",
        type=int,
        default=10,
        help="Window size for rolling average",
    )

    args = parser.parse_args()

    visualize_multi_seed(
        args.json_files, args.json_pattern, args.output_dir, args.rolling_window
    )


if __name__ == "__main__":
    main()
