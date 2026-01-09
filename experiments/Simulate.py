import numpy as np
from tqdm import tqdm
from environment.Environment import Environment
from analysis.Metrics import regret

def simulate(bandit_factory, policy, runs, steps, desc):
    rewards = np.zeros((runs, steps), dtype=float)
    optimal_actions = np.zeros((runs, steps), dtype=int)
    regrets = np.zeros((runs, steps), dtype=float)

    sample_bandit = bandit_factory()
    n_arms = sample_bandit.n_arms
    estimates_history = np.zeros((runs, steps, n_arms), dtype=float)
    counts_history = np.zeros((runs, steps, n_arms), dtype=float)

    run_iter = range(runs)
    run_iter = tqdm(run_iter, f"Simulating {desc}", unit="run")
    
    for run in run_iter:
        bandit = bandit_factory()
        optimal_arm = bandit.optimal_arm
        optimal_reward = bandit.get_optimal_reward()
        
        env = Environment(bandit)
        policy.reset(bandit.n_arms)

        for step in range(steps):
            arm = policy.select_arm()
            reward = env.step(arm)
            policy.update(arm, reward)
            rewards[run, step] = reward
            optimal_actions[run, step] = 1 if arm == optimal_arm else 0
            estimates_history[run, step, :] = policy.value_estimates.copy()
            counts_history[run, step, :] = policy.counts.copy()

        regrets[run, :] = regret(rewards[run, :], optimal_reward)

    result = {
        "avg_reward": rewards.mean(axis=0),
        "optimal_pct": optimal_actions.mean(axis=0) * 100.0,
        "avg_regret": regrets.mean(axis=0),
        "estimates_history": estimates_history,
        "counts_history": counts_history,
    }

    return result

