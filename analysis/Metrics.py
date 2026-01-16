import numpy as np
from matplotlib import pyplot as plt

def cumulative_reward(rewards):
    return np.cumsum(rewards)

def regret(rewards, optimal_reward):
    return np.cumsum(optimal_reward - rewards)

def plot_confidence_intervals(ax, estimates, counts, true_values=None, confidence=1.96):
    n_arms = len(estimates)
    arms = np.arange(1, n_arms + 1)
    std_errors = np.where(counts > 0, 1.0 / np.sqrt(counts), 0)
    conf_intervals = confidence * std_errors
    
    ax.errorbar(arms, estimates, yerr=conf_intervals,
                fmt='o', color='red', markersize=8, capsize=5,
                capthick=1.5, ecolor='gray', elinewidth=1.5,
                label='Estimated Mean')
    
    if true_values is not None:
        ax.scatter(arms, true_values, marker='o', s=80, 
                   facecolors='none', edgecolors='gray', linewidths=1.5,
                   label='True Mean', zorder=5)
    
    ax.set_xlabel('Arm')
    ax.set_ylabel('Value')
    ax.set_xticks(arms)
    ax.legend(loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_metrics(results_by_policy, output_path,
                 estimates_history=None, counts_history=None,
                 true_values=None):
    plt.style.use('fivethirtyeight')
    
    has_ci_data = estimates_history is not None and counts_history is not None
    n_plots = 4 if has_ci_data else 3
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    fig.subplots_adjust(hspace=0.35)

    metric_specs = [
        ("avg_reward", "Average Reward"),
        ("optimal_pct", "% Optimal Action"),
        ("avg_regret", "Cumulative Regret"),
    ]

    for i, (metric_key, ylabel) in enumerate(metric_specs):
        ax = axes[i]
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
    
    axes[2].set_xlabel("Steps")

    if has_ci_data:
        first_policy = list(estimates_history.keys())[0]
        last_step = estimates_history[first_policy].shape[1] - 1
        
        est = estimates_history[first_policy][:, last_step, :].mean(axis=0)
        cnt = counts_history[first_policy][:, last_step, :].mean(axis=0)
        
        plot_confidence_intervals(axes[3], est, cnt, true_values)

    fig.suptitle("Metrics", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
