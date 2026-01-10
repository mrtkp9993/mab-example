from .Policy import Policy
import numpy as np

class ThompsonSamplingContinousRewardsPolicy(Policy):
    def __init__(self, prior_mu=0.0, prior_var=1.0, reward_var=1.0):
        self.prior_mu = float(prior_mu)
        self.prior_var = float(prior_var)
        self.reward_var = float(reward_var)

        if self.prior_var <= 0 or self.reward_var <= 0:
            raise ValueError("prior_var and reward_var must be positive.")

        self.mu = None
        self.precision = None
        self.counts = None
        self.value_estimates = None

    def reset(self, n_arms: int):
        self.mu = np.full(n_arms, self.prior_mu, dtype=float)
        self.precision = np.full(n_arms, 1.0 / self.prior_var, dtype=float)
        self.counts = np.zeros(n_arms, dtype=int)
        self.value_estimates = self.mu.copy()

    def select_arm(self) -> int:
        std = np.sqrt(1.0 / self.precision)
        samples = np.random.normal(self.mu, std)
        max_val = samples.max()
        candidates = np.flatnonzero(samples == max_val)
        return int(np.random.choice(candidates))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1

        obs_precision = 1.0 / self.reward_var
        new_precision = self.precision[arm] + obs_precision
        new_mu = (self.precision[arm] * self.mu[arm] + obs_precision * reward) / new_precision

        self.precision[arm] = new_precision
        self.mu[arm] = new_mu
        self.value_estimates[arm] = new_mu

    def __str__(self):
        return (f"ThompsonSamplingGaussianKnownVariance("
                f"prior_mu={self.prior_mu}, prior_var={self.prior_var}, reward_var={self.reward_var})")
