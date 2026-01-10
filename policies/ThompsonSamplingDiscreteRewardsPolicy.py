from .Policy import Policy
import numpy as np

class ThompsonSamplingDiscreteRewardsPolicy(Policy):
    def __init__(self, prior_alpha=1.0, prior_beta=1.0):
        self.alpha = None
        self.beta = None
        self.value_estimates = None
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def reset(self, n_arms):
        self.alpha = np.full(n_arms, self.prior_alpha, dtype=float)
        self.beta = np.full(n_arms, self.prior_beta, dtype=float)
        self.value_estimates = self.alpha / (self.alpha + self.beta)

    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        max_val = samples.max()
        candidates = np.flatnonzero(samples == max_val)
        return np.random.choice(candidates)

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
        self.value_estimates[arm] = self.alpha[arm] / (self.alpha[arm] + self.beta[arm])

    def __str__(self):
        return f"ThompsonSamplingDiscreteRewardsPolicy(prior_alpha={self.prior_alpha}, prior_beta={self.prior_beta})"
