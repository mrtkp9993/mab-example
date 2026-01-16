from abc import ABC, abstractmethod


class ContextualPolicy(ABC):
    @abstractmethod
    def reset(self, n_arms, context_dim):
        pass
    
    @abstractmethod
    def select_arm(self, context):
        pass
    
    @abstractmethod
    def update(self, arm, reward, context):
        pass
