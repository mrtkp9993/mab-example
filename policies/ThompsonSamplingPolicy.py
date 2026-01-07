from Policy import Policy
import numpy as np

class ThompsonSamplingPolicy(Policy):
    def __init__(self):
        self.alpha = None
        self.beta = None

    # TODO: Implement Thompson Sampling
    def select_arm(self, value_estimates, counts, total_pulls):
        pass