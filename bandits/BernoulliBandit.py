from .Bandit import Bandit
import numpy as np

class BernoulliBandit(Bandit):
    def __init__(self, n_arms, reward_probs):
        super().__init__()
        self.n_arms = n_arms
        if len(reward_probs) != n_arms:
            raise ValueError("reward_probs must have the same length as n_arms")
        self.reward_probs = reward_probs
        self.optimal_arm = np.argmax(reward_probs)

    def pull(self, arm):
        return np.random.binomial(1, self.reward_probs[arm])

    def get_optimal_reward(self):
        return self.reward_probs[self.optimal_arm]

    def __str__(self):
        return f"BernoulliBandit(n_arms={self.n_arms}, reward_probs={self.reward_probs})"
