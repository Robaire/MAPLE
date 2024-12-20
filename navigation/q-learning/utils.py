import numpy as np
import random
import pickle

def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable

def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state][:])
    return action

def epsilon_greedy_policy(Qtable, env, state, epsilon):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action = greedy_policy(Qtable, state)
    else:
        action = env.action_space.sample()

    return action

# Function to save Q-table to a file
def save_qtable(filename, qtable):
    with open(filename, 'wb') as f:
        pickle.dump(qtable, f)
    print(f"Q-table saved to {filename}")

# Function to load Q-table from a file
def load_qtable(filename):
    with open(filename, 'rb') as f:
        qtable = pickle.load(f)
    print(f"Q-table loaded from {filename}")
    return qtable