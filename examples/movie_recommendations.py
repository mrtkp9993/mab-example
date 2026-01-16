"""
Movie Recommendations Example

- K Movies (arms)
- For visitor t, recommend movie a_t ∈ {1,2,...,K}
- Observe reward r_t ∈ {1,2,3,4,5} (rating)
- Goal: Maximize total/average rating of recommendations
"""

import os
import numpy as np

from analysis.Metrics import plot_metrics
from bandits.GaussianBandit import GaussianBandit
from experiments.Simulate import simulate
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from policies.UCB1Policy import UCB1Policy
from policies.AdaptiveGreedyPolicy import AdaptiveGreedyPolicy

# True movie ratings
TRUE_REWARD_MEANS = [3.2, 3.8, 4.5, 3.5, 4.2, 2.8, 3.9, 4.0]
REWARD_STDS = [0.8, 1.0, 0.6, 0.9, 0.7, 1.5, 0.5, 0.7]


class MovieBandit(GaussianBandit):    
    def __init__(self, n_arms, reward_means, reward_stds):
        super().__init__(n_arms, reward_means, reward_stds)
    
    def pull(self, arm):
        raw_reward = super().pull(arm)
        return np.clip(raw_reward, 1.0, 5.0)


def bandit_factory():
    return MovieBandit(
        n_arms=8,
        reward_means=TRUE_REWARD_MEANS,
        reward_stds=REWARD_STDS
    )


def main():
    np.random.seed(42)

    n_visitors = 1000  
    n_runs = 500 

    policies = {
        "Random": EpsilonGreedyPolicy(1.0),              
        "Greedy": EpsilonGreedyPolicy(0.0),              
        "ε-Greedy (0.1)": EpsilonGreedyPolicy(0.1),    
        "UCB1": UCB1Policy(),                            
        "Adaptive Greedy": AdaptiveGreedyPolicy(),      
    }

    print("=" * 60)
    print("Movie Recommendations - Multi-Armed Bandit Simulation")
    print("=" * 60)
    print(f"Number of movies (arms): 8")
    print("Movie catalog:")
    movie_names = [
        "Movie 1", "Movie 2", "Movie 3", "Movie 4", 
        "Movie 5", "Movie 6", "Movie 7", "Movie 8"
    ]
    for i, (name, mean) in enumerate(zip(movie_names, TRUE_REWARD_MEANS), 1):
        print(f"  Movie {i}: {name:20s} (avg rating: {mean})")
    print(f"\nVisitors per run: {n_visitors}")
    print(f"Number of runs: {n_runs}")
    print("=" * 60)

    results = {}
    estimates_history = {}
    counts_history = {}
    
    for label, policy in policies.items():
        result = simulate(
            bandit_factory, 
            policy, 
            runs=n_runs, 
            steps=n_visitors, 
            desc=label
        )
        results[label] = result
        estimates_history[label] = result["estimates_history"]
        counts_history[label] = result["counts_history"]

    # Calculate summary statistics
    print("\n" + "=" * 60)
    print("Results Summary (User Satisfaction)")
    print("=" * 60)
    for label, metrics in results.items():
        avg_rating = metrics["avg_reward"].mean()
        final_rating = metrics["avg_reward"][-50:].mean()
        optimal_pct = metrics["optimal_pct"].mean()
        print(f"{label:20s}: Avg rating {avg_rating:.3f}, "
              f"Final rating {final_rating:.3f}, "
              f"{optimal_pct:.1f}% optimal")

    output_path = os.path.join(".", "movie_recommendations_metrics.png")
    plot_metrics(
        results, 
        output_path,
        estimates_history=estimates_history,
        counts_history=counts_history,
        true_values=np.array(TRUE_REWARD_MEANS)
    )
    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
