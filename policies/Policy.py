from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def select_arm(self, value_estimates, counts, total_pulls):
        pass