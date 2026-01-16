from .Bandit import Bandit
import numpy as np

class GaussianBandit(Bandit):
    def __init__(self, n_arms, reward_means, reward_stds):
        super().__init__()
        self.n_arms = n_arms
        if len(reward_means) != n_arms:
            raise ValueError("reward_means must have the same length as n_arms")
        if len(reward_stds) != n_arms:
            raise ValueError("reward_stds must have the same length as n_arms")
        self.reward_means = reward_means
        self.reward_stds = reward_stds
        self.optimal_arm = np.argmax(reward_means)

    def pull(self, arm):
        return np.random.normal(self.reward_means[arm], self.reward_stds[arm])

    def get_optimal_reward(self):
        return self.reward_means[self.optimal_arm]

    def __str__(self):
        return f"GaussianBandit(n_arms={self.n_arms}, reward_means={self.reward_means}, reward_stds={self.reward_stds})"

class GaussianBanditFactory:
    def __init__(self, n_arms, mean_mean=0, mean_std=1, reward_std=1):
        self.n_arms = n_arms
        self.mean_mean = mean_mean
        self.mean_std = mean_std
        self.reward_std = reward_std
    
    def __call__(self):
        reward_means = np.random.normal(self.mean_mean, self.mean_std, self.n_arms)
        reward_stds = np.full(self.n_arms, self.reward_std)
        return GaussianBandit(self.n_arms, reward_means, reward_stds)
