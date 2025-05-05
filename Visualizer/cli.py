import argparse
import sys

from .parser import parse_output, save_json
from .visualizer import visualize_results
from .multi_seed import visualize_multi_seed


def main():
    """Main entry point for the CLI tool"""
    parser = argparse.ArgumentParser(
        description="RL Visualizer - Analyze and visualize reinforcement learning results",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    parse_parser = subparsers.add_parser(
        "parse", help="Convert RL training output text files to JSON"
    )
    parse_parser.add_argument("input_file", help="Input output file path")
    parse_parser.add_argument(
        "--output", "-o", default=None, help="Output JSON file path"
    )

    vis_parser = subparsers.add_parser(
        "visualize", help="Visualize RL training results from JSON"
    )
    vis_parser.add_argument("json_file", help="Path to the JSON results file")
    vis_parser.add_argument(
        "--output_dir", "-o", default="./plots", help="Directory to save plots"
    )
    vis_parser.add_argument(
        "--rolling_window",
        "-w",
        type=int,
        default=10,
        help="Window size for rolling average",
    )

    multi_parser = subparsers.add_parser(
        "multi", help="Visualize mean RL results across multiple seeds"
    )
    multi_parser.add_argument(
        "--json_files", nargs="+", help="Paths to the JSON results files"
    )
    multi_parser.add_argument(
        "--json_pattern",
        help="Glob pattern to match JSON files (e.g., 'logs/TD3_Pendulum-v1_seed-*.json')",
    )
    multi_parser.add_argument(
        "--output_dir", "-o", default="./plots", help="Directory to save plots"
    )
    multi_parser.add_argument(
        "--rolling_window",
        "-w",
        type=int,
        default=10,
        help="Window size for rolling average",
    )

    args = parser.parse_args()

    if args.command == "parse":
        data = parse_output(args.input_file)
        output_file = args.output
        if output_file is None:
            output_file = args.input_file + ".json"
        save_json(data, output_file)
        print(f"Parsed data saved to {output_file}")

    elif args.command == "visualize":
        visualize_results(args.json_file, args.output_dir, args.rolling_window)

    elif args.command == "multi":
        visualize_multi_seed(
            args.json_files, args.json_pattern, args.output_dir, args.rolling_window
        )

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
