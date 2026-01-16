import numpy as np


class ContextualEnvironment:
    def __init__(self, bandit, context_generator=None):
        self.bandit = bandit
        self.n_arms = bandit.n_arms
        self.context_dim = bandit.context_dim
        self.t = 0
        self.current_context = None
        
        if context_generator is None:
            self.context_generator = lambda: np.random.randn(self.context_dim)
        else:
            self.context_generator = context_generator
    
    def reset(self):
        self.t = 0
        self.current_context = None
    
    def get_context(self):
        self.current_context = self.context_generator()
        self.bandit.set_context(self.current_context)
        return self.current_context
    
    def step(self, arm):
        reward = self.bandit.pull(arm)
        self.t += 1
        return reward
    
    def optimal_reward(self):
        return self.bandit.get_optimal_reward()
    
    def optimal_arm(self):
        return self.bandit.optimal_arm
    
    def __str__(self):
        return f"ContextualEnvironment(bandit={self.bandit})"
