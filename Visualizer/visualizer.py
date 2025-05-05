import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def load_data(json_path):
    """
    Load data from a JSON file

    Args:
        json_path (str): Path to the JSON file

    Returns:
        dict: The loaded data
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def plot_episode_rewards(data, save_path=None, show=True):
    """
    Plot episode rewards over time

    Args:
        data (dict): The experiment data
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
    """
    episode_nums = [e["episode_num"] for e in data["episodes"]]
    rewards = [e["reward"] for e in data["episodes"]]

    plt.figure(figsize=(10, 6))
    plt.plot(episode_nums, rewards)
    plt.title(
        f"Episode Rewards - {data['experiment']['policy']} on {data['experiment']['environment']}"
    )
    plt.xlabel("Episode Number")
    plt.ylabel("Total Reward")
    plt.grid(True, linestyle="--", alpha=0.7)

    if len(rewards) > 1:
        z = np.polyfit(episode_nums, rewards, 1)
        p = np.poly1d(z)
        plt.plot(
            episode_nums,
            p(episode_nums),
            "r--",
            alpha=0.8,
            label=f"Trend: {z[0]:.2f}x + {z[1]:.2f}",
        )
        plt.legend()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_evaluation_performance(data, save_path=None, show=True):
    """
    Plot evaluation performance over timesteps

    Args:
        data (dict): The experiment data
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
    """
    if not data["evaluations"]:
        print("No evaluation data available")
        return

    timesteps = [e["at_timesteps"] for e in data["evaluations"]]

    eval_key = [
        k for k in data["evaluations"][0].keys() if k.startswith("evaluation_over")
    ][0]
    eval_scores = [e[eval_key] for e in data["evaluations"]]

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, eval_scores, "o-")
    plt.title(
        f"Evaluation Performance - {data['experiment']['policy']} on {data['experiment']['environment']}"
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.grid(True, linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_rolling_rewards(data, window=10, save_path=None, show=True):
    """
    Plot rolling average of episode rewards

    Args:
        data (dict): The experiment data
        window (int): Size of the rolling window
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
    """
    episode_nums = [e["episode_num"] for e in data["episodes"]]
    rewards = [e["reward"] for e in data["episodes"]]

    rolling_rewards = []
    for i in range(len(rewards)):
        if i < window - 1:
            rolling_rewards.append(np.mean(rewards[: i + 1]))
        else:
            rolling_rewards.append(np.mean(rewards[i - window + 1 : i + 1]))

    plt.figure(figsize=(10, 6))
    plt.plot(episode_nums, rewards, "b-", alpha=0.3, label="Episode Rewards")
    plt.plot(
        episode_nums, rolling_rewards, "r-", label=f"{window}-Episode Rolling Average"
    )
    plt.title(
        f"Episode Rewards - {data['experiment']['policy']} on {data['experiment']['environment']}"
    )
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def visualize_results(json_path, output_dir="./plots", rolling_window=10):
    """
    Visualize all results from a single JSON file

    Args:
        json_path (str): Path to the JSON file
        output_dir (str): Directory to save plots
        rolling_window (int): Window size for rolling average
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = load_data(json_path)

    base_name = os.path.splitext(os.path.basename(json_path))[0]

    plot_episode_rewards(
        data, save_path=os.path.join(output_dir, f"{base_name}_rewards.png")
    )
    plot_evaluation_performance(
        data, save_path=os.path.join(output_dir, f"{base_name}_eval.png")
    )
    plot_rolling_rewards(
        data,
        window=rolling_window,
        save_path=os.path.join(output_dir, f"{base_name}_rolling_{rolling_window}.png"),
    )

    print(f"Plots saved to {output_dir}")


def main():
    """Command-line interface for visualizing results"""
    parser = argparse.ArgumentParser(
        description="Visualize RL training results from JSON"
    )
    parser.add_argument("json_file", help="Path to the JSON results file")
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
    visualize_results(args.json_file, args.output_dir, args.rolling_window)


if __name__ == "__main__":
    main()
