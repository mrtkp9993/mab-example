from .Policy import Policy
import numpy as np

class UCB1Policy(Policy):
    def __init__(self, c=2.0):
        self.counts = None
        self.value_estimates = None
        self.c = c

    def reset(self, n_arms):
        self.counts = np.zeros(n_arms, dtype=int)
        self.value_estimates = np.zeros(n_arms, dtype=float)

    def select_arm(self):
        total_pulls = np.sum(self.counts)
        if total_pulls < len(self.counts):
            return total_pulls
        conf = self.c * np.sqrt(np.log(total_pulls) / self.counts)
        return np.argmax(self.value_estimates+conf)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.value_estimates[arm] += (reward - self.value_estimates[arm]) / self.counts[arm]

    def __str__(self):
        return f"UCB1Policy(c={self.c})"
