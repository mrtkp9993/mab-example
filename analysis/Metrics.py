import numpy as np
from matplotlib import pyplot as plt

def cumulative_reward(rewards):
    return np.cumsum(rewards)

def regret(rewards, optimal_reward):
    return np.cumsum(optimal_reward - rewards)

def plot_metrics(results_by_policy, output_path):
    plt.style.use('fivethirtyeight')
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    metric_specs = [
        ("avg_reward", "Average Reward"),
        ("optimal_pct", "% Optimal Action"),
        ("avg_regret", "Cumulative Regret"),
    ]

    for ax, (metric_key, ylabel) in zip(axes, metric_specs):
        for label, metrics in results_by_policy.items():
            ax.plot(
                metrics[metric_key], 
                label=label, 
                linewidth=1.5,
                alpha=0.9
            )
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', framealpha=0.95, edgecolor='none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel("Steps")
    fig.suptitle("Metrics")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
