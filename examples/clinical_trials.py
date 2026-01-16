"""
Clinical Trials

- K = 4 Arms (Treatments)
- For patient t, prescribe treatment a_t ∈ {1,2,3,4}
- Observe reward r_t ∈ {0, 1} (1 if healed, 0 if not)
- Goal: Maximize total number of patients healed
"""

import os
import numpy as np

from analysis.Metrics import plot_metrics
from bandits.BernoulliBandit import BernoulliBandit
from experiments.Simulate import simulate
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from policies.UCB1Policy import UCB1Policy

# True treatment success rates
TRUE_REWARD_PROBS = [0.30, 0.50, 0.70, 0.45]


def bandit_factory():
    return BernoulliBandit(
        n_arms=4,
        reward_probs=TRUE_REWARD_PROBS
    )


def main():
    np.random.seed(42)

    n_patients = 500
    n_trials = 1000

    policies = {
        "Random (ε=1.0)": EpsilonGreedyPolicy(1.0),
        "Greedy (ε=0.0)": EpsilonGreedyPolicy(0.0),
        "ε-Greedy (ε=0.1)": EpsilonGreedyPolicy(0.1),
        "UCB1": UCB1Policy(),
    }

    print("=" * 60)
    print("Clinical Trials - Multi-Armed Bandit Simulation")
    print("=" * 60)
    print(f"Number of treatments (arms): 4")
    print(f"Treatment success rates: [30%, 50%, 70%, 45%]")
    print(f"Patients per trial: {n_patients}")
    print(f"Number of trials: {n_trials}")
    print("=" * 60)

    results = {}
    estimates_history = {}
    counts_history = {}
    
    for label, policy in policies.items():
        result = simulate(
            bandit_factory, 
            policy, 
            runs=n_trials, 
            steps=n_patients, 
            desc=label
        )
        results[label] = result
        estimates_history[label] = result["estimates_history"]
        counts_history[label] = result["counts_history"]

    # Calculate summary statistics
    print("\n" + "=" * 60)
    print("Results Summary (Patients Healed)")
    print("=" * 60)
    for label, metrics in results.items():
        total_healed = metrics["avg_reward"].sum()
        healing_rate = metrics["avg_reward"].mean() * 100
        optimal_pct = metrics["optimal_pct"].mean()
        print(f"{label:25s}: {total_healed:6.1f} healed ({healing_rate:.1f}% rate), "
              f"{optimal_pct:.1f}% optimal action")

    output_path = os.path.join(".", "clinical_trials_metrics.png")
    plot_metrics(
        results, 
        output_path,
        estimates_history=estimates_history,
        counts_history=counts_history,
        true_values=np.array(TRUE_REWARD_PROBS)
    )
    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
