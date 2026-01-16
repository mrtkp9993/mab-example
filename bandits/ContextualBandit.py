from .Bandit import Bandit
import numpy as np


class ContextualBandit(Bandit):
    def __init__(self, n_arms, context_dim, theta=None, noise_std=0.1):
        super().__init__()
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.noise_std = noise_std
        
        if theta is not None:
            self.theta = np.asarray(theta)
            if self.theta.shape != (n_arms, context_dim):
                raise ValueError(f"theta shape must be ({n_arms}, {context_dim})")
        else:
            self.theta = np.random.randn(n_arms, context_dim)
            self.theta /= np.linalg.norm(self.theta, axis=1, keepdims=True)
        
        self.current_context = None
        self.optimal_arm = None
    
    def set_context(self, context):
        self.current_context = np.asarray(context).flatten()
        expected_rewards = self.theta @ self.current_context
        self.optimal_arm = np.argmax(expected_rewards)
    
    def pull(self, arm):
        if self.current_context is None:
            raise ValueError("Context must be set before pulling an arm")
        
        expected_reward = np.dot(self.theta[arm], self.current_context)
        noise = np.random.normal(0, self.noise_std)
        return expected_reward + noise
    
    def get_optimal_reward(self):
        if self.current_context is None:
            raise ValueError("Context must be set before getting optimal reward")
        return np.dot(self.theta[self.optimal_arm], self.current_context)
    
    def get_expected_rewards(self, context=None):
        ctx = context if context is not None else self.current_context
        return self.theta @ ctx
    
    def __str__(self):
        return f"ContextualBandit(n_arms={self.n_arms}, context_dim={self.context_dim})"
