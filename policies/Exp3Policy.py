from .Policy import Policy
import numpy as np


class Exp3Policy(Policy):
    def __init__(self, gamma=0.1):
        self.gamma = gamma
        self.n_arms = None
        self.weights = None
        self.counts = None
        self.value_estimates = None
    
    def reset(self, n_arms):
        self.n_arms = n_arms
        self.weights = np.ones(n_arms, dtype=float)
        self.counts = np.zeros(n_arms, dtype=int)
        self.value_estimates = np.zeros(n_arms, dtype=float)
    
    def _get_probabilities(self):
        total_weight = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / total_weight) + \
                (self.gamma / self.n_arms)
        return probs
    
    def select_arm(self):
        probs = self._get_probabilities()
        return np.random.choice(self.n_arms, p=probs)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        probs = self._get_probabilities()
        reward_clipped = np.clip(reward, 0, 1)
        estimated_reward = reward_clipped / probs[arm]
        self.weights[arm] *= np.exp(self.gamma * estimated_reward / self.n_arms)
        self.weights /= np.max(self.weights)
        n = self.counts[arm]
        self.value_estimates[arm] += (reward - self.value_estimates[arm]) / n
    
    def __str__(self):
        return f"Exp3Policy(gamma={self.gamma})"
