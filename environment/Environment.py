class Environment:
    def __init__(self, bandit):
        self.bandit = bandit
        self.n_arms = bandit.n_arms
        self.t = 0

    def reset(self):
        self.t = 0

    def step(self, arm):
        reward = self.bandit.pull(arm)
        self.t += 1
        return reward

    def optimal_reward(self):
        return self.bandit.get_optimal_reward()

    def __str__(self):
        return f"Environment(bandit={self.bandit})"