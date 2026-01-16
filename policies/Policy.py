from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def reset(self, n_arms):
        pass

    @abstractmethod
    def select_arm(self):
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass