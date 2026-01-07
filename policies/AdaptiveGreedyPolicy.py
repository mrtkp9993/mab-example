from Policy import Policy
import numpy as np

class AdaptiveGreedyPolicy(Policy):
    def __init__(self, c):
        self.c = c

    def select_arm(self, counts, value_estimates, total_pulls):
        epsilon = self.c / (self.c + total_pulls) if total_pulls > 0 else 1.0
        if np.random.rand() < epsilon:
            return np.random.randint(len(value_estimates))
        return np.argmax(value_estimates)

    def __str__(self):
        return f"AdaptiveGreedyPolicy(c={self.c})"