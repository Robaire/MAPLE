# Rover Charging Simulator

This repository contains a simulation environment for testing rover charging agents. The simulation allows you to evaluate how effectively your rover agent can locate and connect to charging stations from various starting positions.

## Overview

The repository includes a battery evaluation system that:
1. Spawns a rover at different coordinates specified in `coordinates.txt`
2. Runs your charging agent implementation
3. Records success/failure results in `charging_results.csv`

## How It Works

The `battery_evaluator.sh` script automatically tests your charging agent by:
- Reading spawn locations from `coordinates.txt`
- Running your agent from each location
- Tracking whether the rover successfully locates and connects to the charging station
- Saving results to `charging_results.csv` for analysis

## Getting Started

### Prerequisites
- Python environment (using uv)
- Simulator package installed

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rover-charging-simulator.git
cd rover-charging-simulator
```

2. Configure spawn locations:
Edit `coordinates.txt` to define the starting positions for testing. Each line should contain x,y coordinates.

Example `coordinates.txt`:
```
0,0
10,15
-5,20
```

3. **Important**: Modify your charging agent implementation to print the success message:
```python
# Make sure your agent prints this exact message when it successfully connects to a charging station
print(f'CHARGING NOTIFICATION FROM SIMULATOR')
```

### Running Tests

1. Make sure your agent implementation is located at `agents/charging_agent.py`

2. **Important**: Modify the `battery_evaluator.sh` script to use your agent:
```bash
# Find this line in battery_evaluator.sh and make sure it points to your agent
output=$(timeout $TIMEOUT_PER_RUN uv run ./scripts/run_agent.py agents/charging_agent.py --sim="./simulator" --xy="[$x, $y]" 2>&1)
```

3. Run the evaluation script:
```bash
./battery_evaluator.sh
```

4. Check results in `charging_results.csv`

## Results

The `charging_results.csv` file will contain the evaluation results with the following format:
- Starting coordinates
- Success/failure status
- Time taken (if successful)
- Any error messages (if failed)

## Troubleshooting

If your agent isn't being properly recognized:
1. Verify your agent is printing `CHARGING NOTIFICATION FROM SIMULATOR` when it successfully connects
2. Check that you've correctly modified the `battery_evaluator.sh` script to run your agent
3. Ensure your agent implementation is in the correct location

## Contributing

Feel free to submit pull requests with improvements to the evaluation system or example agents.

## License

This project is licensed under the MIT License - see the LICENSE file for details.