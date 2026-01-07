from Policy import Policy
import numpy as np

class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_arm(self, counts, value_estimates, total_pulls):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(value_estimates))
        return np.argmax(value_estimates)

    def __str__(self):
        return f"EpsilonGreedyPolicy(epsilon={self.epsilon})"