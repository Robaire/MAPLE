import argparse
import gymnasium as gym
import custom_env as custom_env
import numpy as np
import yaml

from tqdm import tqdm
from evaluate import evaluate_agent
from record import record_video
from utils import initialize_q_table, epsilon_greedy_policy, save_qtable, load_qtable


def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
        state, info = env.reset()
        stop = 0
        terminated = False
        truncated = False

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, env, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

            if terminated or truncated:
                break

            state = new_state

    return Qtable

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement Learning Script")
    parser.add_argument("--env", choices=["FrozenLake-v1", "Taxi-v3", "Custom"], required=True, help="Choose the environment")
    parser.add_argument("--config", choices=["config/taxi.yml", "config/frozenlake.yml", "config/custom.yml"], required=True, help="Choose the configuration file")
    args = parser.parse_args()

    # Load configuration from the selected YAML file
    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    n_training_episodes = config["n_training_episodes"]
    learning_rate = config["learning_rate"]
    n_eval_episodes = config["n_eval_episodes"]
    max_steps = config["max_steps"]
    gamma = config["gamma"]
    max_epsilon = config["max_epsilon"]
    min_epsilon = config["min_epsilon"]
    decay_rate = config["decay_rate"]
    video_out_dir = config["video_out_dir"]
    eval_seed = config["eval_seed"]

    if args.env == "FrozenLake-v1":
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")
    elif args.env == "Taxi-v3":
        env = gym.make("Taxi-v3", render_mode="rgb_array")
    elif args.env == "Custom":
        env = custom_env.Custom()

    state_space = env.observation_space.n
    action_space = env.action_space.n

    Qtable_frozenlake = initialize_q_table(state_space, action_space)
    # Code to load qtable
    # Qtable_frozenlake = load_qtable("qtable test")
    Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)

    # Code to save qtable
    # IMPORTANT NOTE: Make sure not to override any previous ones, change the name if needed
    # save_qtable("qtable test", Qtable_frozenlake)

    mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    record_video(env, Qtable_frozenlake, out_directory=video_out_dir, fps=1)
