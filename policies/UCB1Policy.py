from Policy import Policy
import numpy as np

class UCB1Policy(Policy):
    def __init__(self, c=2.0):
        self.c = c

    def select_arm(self, counts, value_estimates, total_pulls):
        if total_pulls < len(counts):
            return total_pulls
        conf = self.c * np.sqrt(np.log(total_pulls) / counts)
        return np.argmax(value_estimates+conf)

    def __str__(self):
        return f"UCB1Policy(c={self.c})"