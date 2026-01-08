from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def select_arm(self, counts, value_estimates, total_pulls):
        pass
