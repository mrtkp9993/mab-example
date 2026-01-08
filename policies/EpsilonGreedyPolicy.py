from .Policy import Policy
import numpy as np

class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        self.counts = None
        self.value_estimates = None
        self.epsilon = epsilon

    def reset(self, n_arms):
        self.counts = np.zeros(n_arms, dtype=int)
        self.value_estimates = np.zeros(n_arms, dtype=float)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.value_estimates))
        max_q = np.max(self.value_estimates)
        max_actions = np.where(self.value_estimates == max_q)[0]
        return np.random.choice(max_actions)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.value_estimates[arm] += (reward - self.value_estimates[arm]) / self.counts[arm]

    def __str__(self):
        return f"EpsilonGreedyPolicy(epsilon={self.epsilon})"
