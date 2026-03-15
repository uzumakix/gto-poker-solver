"""
solve.py -- Run CFR+ and generate convergence + strategy visuals
================================================================

Trains a CFR+ solver on Kuhn Poker, prints the resulting Nash
equilibrium strategies, and saves:
    results/convergence.png   -- log-scale exploitability over iterations
    results/strategy_heatmap.png -- action probabilities per info set
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from cfr import CFRSolver

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ITERATIONS = 50_000


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    solver = CFRSolver(use_cfr_plus=True)
    print(f"Training CFR+ for {ITERATIONS:,d} iterations...")
    history = solver.train(ITERATIONS)

    # print final strategies
    print("\n--- Average Strategy Profile ---")
    table = solver.get_strategy_table()
    for key, strat in table.items():
        print(f"  {key:8s}  pass={strat[0]:.4f}  bet={strat[1]:.4f}")

    gv = solver.game_value()
    eps = solver.exploitability()
    print(f"\nGame value (p0): {gv:.6f}  (Nash = -0.055556)")
    print(f"Exploitability:  {eps:.6f}")

    # convergence plot
    plot_convergence(history)

    # strategy heatmap
    plot_heatmap(table)

    print(f"\nPlots saved to {RESULTS_DIR}/")


def plot_convergence(history):
    """Log-scale exploitability vs iteration count."""
    iters = [h[0] for h in history]
    eps = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, eps, color="#2563eb", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("Exploitability", fontsize=12)
    ax.set_title("CFR+ Convergence on Kuhn Poker", fontsize=14)
    ax.grid(True, alpha=0.3, which="both")

    # annotate final value
    final_eps = eps[-1]
    ax.axhline(y=final_eps, color="#dc2626", linestyle="--",
               alpha=0.5, linewidth=0.8)
    ax.annotate(f"final = {final_eps:.2e}",
                xy=(iters[-1], final_eps),
                xytext=(-80, 15), textcoords="offset points",
                fontsize=9, color="#dc2626")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "convergence.png"), dpi=150)
    plt.close(fig)
    print("Saved convergence.png")


def plot_heatmap(strategy_table):
    """Strategy heatmap: rows = info sets, columns = actions."""
    # separate player 0 and player 1 info sets for cleaner layout
    keys_p0 = []
    keys_p1 = []
    for k in strategy_table:
        parts = k.split()
        if len(parts) == 1:
            # root node (player 0, no history)
            keys_p0.append(k)
        else:
            history = parts[1]
            player = len(history) % 2
            if player == 0:
                keys_p0.append(k)
            else:
                keys_p1.append(k)

    all_keys = keys_p0 + keys_p1
    labels = []
    for k in all_keys:
        parts = k.split()
        card = parts[0]
        hist = parts[1] if len(parts) > 1 else "(root)"
        # indicate which player
        if k in keys_p0:
            labels.append(f"P0 | {card} | {hist}")
        else:
            labels.append(f"P1 | {card} | {hist}")

    data = np.array([strategy_table[k] for k in all_keys])
    col_labels = ["Pass / Check / Fold", "Bet / Call"]

    fig, ax = plt.subplots(figsize=(7, max(5, len(all_keys) * 0.45)))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=col_labels, yticklabels=labels,
        vmin=0, vmax=1, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Action Probability"}, ax=ax,
    )
    ax.set_title("Nash Equilibrium Strategy Profile", fontsize=13)
    ax.tick_params(axis="y", rotation=0)

    # divider between P0 and P1
    ax.axhline(y=len(keys_p0), color="black", linewidth=2)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "strategy_heatmap.png"), dpi=150)
    plt.close(fig)
    print("Saved strategy_heatmap.png")


if __name__ == "__main__":
    main()
