from .ContextualPolicy import ContextualPolicy
import numpy as np


class LinUCBPolicy(ContextualPolicy):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.n_arms = None
        self.context_dim = None
        self.A = None  
        self.b = None
        self.theta = None
        
        self.counts = None
        self.value_estimates = None
    
    def reset(self, n_arms, context_dim):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.theta = [np.zeros(context_dim) for _ in range(n_arms)]
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.value_estimates = np.zeros(n_arms, dtype=float)
    
    def select_arm(self, context):
        context = np.asarray(context).flatten()
        ucb_values = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            theta_a = np.linalg.solve(self.A[a], self.b[a])
            self.theta[a] = theta_a
            A_inv_x = np.linalg.solve(self.A[a], context)
            expected_reward = np.dot(context, theta_a)
            exploration_bonus = self.alpha * np.sqrt(np.dot(context, A_inv_x))
            ucb_values[a] = expected_reward + exploration_bonus
        
        return np.argmax(ucb_values)
    
    def update(self, arm, reward, context):
        context = np.asarray(context).flatten()
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.counts[arm] += 1
        n = self.counts[arm]
        self.value_estimates[arm] += (reward - self.value_estimates[arm]) / n
    
    def get_theta(self, arm):
        return np.linalg.solve(self.A[arm], self.b[arm])
    
    def __str__(self):
        return f"LinUCBPolicy(alpha={self.alpha})"
