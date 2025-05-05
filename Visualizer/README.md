# Result Visualizer

## Features

- [Only for Adrian] Parse text output from RL training into structured JSON format
- Log experiment data with the `RLLogger` class
- Visualize training results with customizable plots
- Compare and average results across multiple seeds

## Usage

### Importing the Module

**For Jupyter Notebooks:**

First cell:

```python
import sys
import os

sys.path.append(os.path.abspath('..'))
```

Second cell:

```python
import Visualizer as rlvis
```

**For Python scripts:**

```python
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import Visualizer as rlvis
```

---

### Parsing Text Output Files

```python
# From Python
data = rlvis.parse_output("TD3_Pendulum-v1_seed-0.txt")
rlvis.save_json(data, "TD3_Pendulum-v1_seed-0.json")

# From command line
python -m Visualizer.cli parse /your/path/to/TD3_Pendulum-v1_seed-0.txt
```

### Logging Experiments

```python
logger = rlvis.RLLogger(policy="TD3", environment="Pendulum-v1", seed=0)

# Log episode data
logger.log_episode(total_timesteps=200, episode_num=1, episode_timesteps=200, reward=-1608.45)

# Log evaluation data
logger.log_evaluation(at_timesteps=5000, eval_episodes=10, avg_reward=-1690.832)

# Save the JSON file
json_path = logger.save()
```

### Visualizing Single-Seed Results

```python
# From Python
data = rlvis.load_data("TD3_Pendulum-v1_seed-0.json")
rlvis.plot_episode_rewards(data)
rlvis.plot_evaluation_performance(data)
rlvis.plot_rolling_rewards(data, window=10)

# Visualize all plots at once
rlvis.visualize_results("TD3_Pendulum-v1_seed-0.json")

# From command line
python -m Visualizer.cli visualize /your/path/to/TD3_Pendulum-v1_seed-0.json
```

### Visualizing Multi-Seed Results

```python
# From Python
json_files = ["TD3_Pendulum-v1_seed-0.json", "TD3_Pendulum-v1_seed-1.json", "TD3_Pendulum-v1_seed-2.json"]
rlvis.visualize_multi_seed(json_files=json_files)

# From command line
python -m Visualizer.cli multi --json_files your/path/to/TD3_Pendulum-v1_seed-0.json your/path/to/TD3_Pendulum-v1_seed-1.json your/path/to/TD3_Pendulum-v1_seed-2.json

# Or using a pattern
python -m Visualizer.cli multi --json_pattern "your/path/to/TD3_Pendulum-v1_seed-*.json"
```

### Using in Jupyter Notebooks

```python
import sys
import os
sys.path.append(os.path.abspath('..'))

import Visualizer as rlvis

# Load data from files
json_files = ["TD3_Pendulum-v1_seed-0.json", "TD3_Pendulum-v1_seed-1.json", "TD3_Pendulum-v1_seed-2.json"]
all_data = [rlvis.load_data(f) for f in json_files]

# Generate plots directly in the notebook
rlvis.plot_individual_seeds(all_data)
rlvis.plot_mean_episode_rewards(all_data)
rlvis.plot_mean_evaluation_performance(all_data)
rlvis.plot_mean_rolling_rewards(all_data, window=10)
```

## Command-Line Interface

The package provides a combined command-line interface:

```bash
# Main help
python -m Visualizer.cli --help

# Parse text output to JSON
python -m Visualizer.cli parse /your/path/to/TD3_Pendulum-v1_seed-0.txt

# Visualize single-seed results
python -m Visualizer.cli visualize /your/path/to/TD3_Pendulum-v1_seed-0.json

# Visualize multi-seed results
python -m Visualizer.cli multi --json_pattern "your/path/to/TD3_Pendulum-v1_seed-*.json"
```

## Data Format

The JSON data format is structured as follows:

```json
{
  "experiment": {
    "policy": "TD3",
    "environment": "Pendulum-v1",
    "seed": 0,
    "start_time": "2023-06-01T12:34:56.789"
  },
  "episodes": [
    {
      "total_timesteps": 200,
      "episode_num": 1,
      "episode_timesteps": 200,
      "reward": -1608.45
    }
    // ... more episodes ...
  ],
  "evaluations": [
    {
      "at_timesteps": 0,
      "evaluation_over_10_episodes": -1690.832
    }
    // ... more evaluations ...
  ]
}
```
