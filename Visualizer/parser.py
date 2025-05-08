import re
import json
import argparse


def parse_output(output_file):
    """
    Args:
        output_file (str): Path to the output file

    Returns:
        dict: Structured data with experiment info, episodes and evaluations
    """
    with open(output_file, "r") as f:
        content = f.read()

    policy_match = re.search(r"Policy: (\w+), Env: ([\w-]+), Seed: (\d+)", content)
    experiment = {
        "policy": policy_match.group(1),
        "environment": policy_match.group(2),
        "seed": int(policy_match.group(3)),
    }

    episode_pattern = (
        r"Total T: (\d+) Episode Num: (\d+) Episode T: (\d+) Reward: (-?\d+\.\d+)"
    )
    episode_matches = re.finditer(episode_pattern, content)

    episodes = []
    for match in episode_matches:
        episodes.append(
            {
                "total_timesteps": int(match.group(1)),
                "episode_num": int(match.group(2)),
                "episode_timesteps": int(match.group(3)),
                "reward": float(match.group(4)),
            }
        )

    eval_pattern = r"Evaluation over (\d+) episodes: (-?\d+\.\d+)"
    eval_matches = re.finditer(eval_pattern, content)

    evaluations = []
    eval_timesteps = []

    for i, match in enumerate(
        re.finditer(r"---------------------------------------\nEvaluation", content)
    ):
        if i > 0:
            pos = match.start()
            episode_before = re.findall(r"Total T: (\d+)", content[:pos])
            if episode_before:
                eval_timesteps.append(int(episode_before[-1]))

    eval_timesteps.insert(0, 0)

    for i, match in enumerate(eval_matches):
        if i < len(eval_timesteps):
            evaluations.append(
                {
                    "at_timesteps": eval_timesteps[i],
                    "evaluation_over_" + match.group(1) + "_episodes": float(
                        match.group(2)
                    ),
                }
            )

    return {"experiment": experiment, "episodes": episodes, "evaluations": evaluations}


def save_json(data, output_file):
    """
    Save the structured data as JSON

    Args:
        data (dict): The data to save
        output_file (str): Output file path
    """
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


def main():
    """Command-line interface for parsing output files"""
    parser = argparse.ArgumentParser(description="Parse RL training output into JSON")
    parser.add_argument("input_file", help="Input output file path")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")

    args = parser.parse_args()

    data = parse_output(args.input_file)

    output_file = args.output
    if output_file is None:
        output_file = args.input_file + ".json"

    save_json(data, output_file)
    print(f"Parsed data saved to {output_file}")


if __name__ == "__main__":
    main()
