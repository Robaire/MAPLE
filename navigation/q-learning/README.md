# Q-Learning in Grid World Environments

This project focuses on implementing and demonstrating the Q-learning algorithm in two classic grid world environments: FrozenLake and TaxiV3. Q-learning is a fundamental reinforcement learning algorithm that allows an agent to learn optimal actions through exploration and exploitation.

## Table of Contents

- [Introduction to Q-Learning](#introduction-to-q-learning)
- [Grid World Environments](#grid-world-environments)
- [Implementation](#implementation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction to Q-Learning

Q-Learning is a model-free, off-policy reinforcement learning algorithm used to find the optimal action-selection policy for a given finite Markov decision process (MDP). It's one of the foundational techniques in reinforcement learning and has been successfully applied to a wide range of problems. Q-learning aims to learn a Q-value function that represents the expected cumulative reward an agent can obtain by taking a certain action in a given state.

## Grid World Environments

This repository includes implementations for two classic grid world environments:

1. **FrozenLake**: A simple grid world where the agent must navigate across a frozen lake while avoiding holes. The agent receives a reward of +1 for reaching the goal state and a reward of 0 otherwise.

2. **TaxiV3**: An environment where the agent controls a taxi navigating through a city grid to pick up and drop off passengers at designated locations. The agent receives rewards based on its actions and progress towards delivering passengers.

## Implementation

The Q-learning algorithm has been implemented in Python using the OpenAI Gym library to interact with the grid world environments. The main components of the implementation include:

Initialization of Q-values for all state-action pairs.
Exploration and exploitation strategy (e.g., epsilon-greedy policy).
Q-value update based on the Bellman equation.
Training loop that allows the agent to interact with the environment and improve its Q-values.

## Usage

1. Clone this repository and install dependencies in your python environment
   ```bash
   # clone repository
   git clone https://github.com/adhiiisetiawan/q-learning-grid-env.git
   cd q-learning-grid-env

   # install dependencies
   pip3 install -r requirements.txt
   ```
2. Set variables
```
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
```

3. Start training <br>
   For frozenlake environment
   ```bash
   python3 train.py --env FrozenLake-v1 --config config/frozenlake.yml
   ```

   For taxi environment
   ```bash
   python3 train.py --env Taxi-v3 --config config/taxi.yml
   ```

## Results
FrozenLake environment results<br>
![FrozenLake Recorded GIF](https://github.com/adhiiisetiawan/q-learning-grid-env/blob/main/results/replay_frozenlake.gif)

Taxi environment results<br>
![Taxi Recorded GIF](https://github.com/adhiiisetiawan/q-learning-grid-env/blob/main/results/replay_taxi.gif)

## License

This project is licensed under the [MIT License](https://github.com/adhiiisetiawan/q-learning-grid-env/blob/main/LICENSE).


## For NASA you can use this command to run our version

python3 train.py --env Custom --config config/custom.yml

If you want to load or save qtable this option is located in train.py

I also set a limit to the saved frames in record.py to help prevent crashing, this will need to be changed if you wish to record

You can change variables in the .yml file

custom_env contains the skeleton that will be used for q learning