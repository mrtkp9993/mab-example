from .Policy import Policy
import numpy as np

class AdaptiveGreedyPolicy(Policy):
    def __init__(self, c=2.0):
        self.counts = None
        self.value_estimates = None
        self.c = c

    def reset(self, n_arms):
        self.counts = np.zeros(n_arms, dtype=int)
        self.value_estimates = np.zeros(n_arms, dtype=float)

    def select_arm(self):
        total_pulls = np.sum(self.counts)
        epsilon = self.c / (self.c + total_pulls) if total_pulls > 0 else 1.0
        if np.random.rand() < epsilon:
            return np.random.randint(len(self.value_estimates))
        return np.argmax(self.value_estimates)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.value_estimates[arm] += (reward - self.value_estimates[arm]) / self.counts[arm]

    def __str__(self):
        return f"AdaptiveGreedyPolicy(c={self.c})"
