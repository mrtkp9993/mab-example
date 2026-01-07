import numpy as np

class Agent:
    def __init__(self, n_arms, policy):
        self.pull_counts = None
        self.value_estimates = None
        self.n_arms = n_arms
        self.policy = policy
        self.total_pulls = 0
        self.reset()

    def reset(self):
        self.pull_counts = np.zeros(self.n_arms, dtype=int)
        self.value_estimates = np.zeros(self.n_arms, dtype=float)
        self.total_pulls = 0

    def select_arm(self):
        return self.policy.select_arm(self.pull_counts, self.value_estimates, self.total_pulls)

    def update(self, arm, reward):
        self.pull_counts[arm] += 1
        self.value_estimates[arm] += (reward - self.value_estimates[arm]) / self.pull_counts[arm]
        self.total_pulls += 1

    def __str__(self):
        return f"Agent(n_arms={self.n_arms}, policy={self.policy}, total_pulls={self.total_pulls}, pull_counts={self.pull_counts}, value_estimates={self.value_estimates})"