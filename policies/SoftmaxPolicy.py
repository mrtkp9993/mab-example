from .Policy import Policy
import numpy as np

class SoftmaxPolicy(Policy):
    def __init__(self, temperature=1.0):
        self.counts = None
        self.value_estimates = None
        self.temperature = temperature

    def reset(self, n_arms):
        self.counts = np.zeros(n_arms, dtype=int)
        self.value_estimates = np.zeros(n_arms, dtype=float)

    def select_arm(self):
        if self.temperature <= 0:
            return np.argmax(self.value_estimates)
        scaled = self.value_estimates / self.temperature
        max_scaled = np.max(scaled)
        exp_values = np.exp(scaled - max_scaled)
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(len(self.value_estimates), p=probabilities)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.value_estimates[arm] += (reward - self.value_estimates[arm]) / self.counts[arm]

    def __str__(self):
        return f"SoftmaxPolicy(temperature={self.temperature})"
