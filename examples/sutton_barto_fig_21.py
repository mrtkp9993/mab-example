import os

import numpy as np

from analysis.Metrics import plot_metrics
from bandits.GaussianBandit import GaussianBanditFactory
from experiments.Simulate import simulate
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy


def main():
    np.random.seed(42)

    n_arms = 10
    runs = 2000
    time_steps = 1000
    
    bandit_factory = GaussianBanditFactory(n_arms)
    
    policies = {
        "Epsilon-Greedy (0.0)": EpsilonGreedyPolicy(0.0),
        "Epsilon-Greedy (0.01)": EpsilonGreedyPolicy(0.01),
        "Epsilon-Greedy (0.1)": EpsilonGreedyPolicy(0.1)
    }
    results = {}
    for label, policy in policies.items():
        results[label] = simulate(bandit_factory, policy, runs, time_steps, desc=label)

    plot_metrics(results, os.path.join(".", "metrics.png"))

if __name__ == "__main__":
    main()
