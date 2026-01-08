import numpy as np

def cumulative_reward(rewards):
    return np.cumsum(rewards)

def regret(rewards, optimal_reward):
    return np.cumsum(optimal_reward - rewards)