"""
solve.py -- Run CFR+ and generate convergence + strategy visuals
================================================================

Trains a CFR+ solver on Kuhn Poker, prints the resulting Nash
equilibrium strategies, and saves:
    results/convergence.png      -- log-scale exploitability over iterations
    results/strategy_heatmap.png -- action probabilities per info set
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


# maps info set key to a human-readable row label and the
# meaning of "action 0" and "action 1" at that decision point
_INFO_SET_LABELS = {
    # P0 root decisions
    "J":    ("P0  J  opening",      "Check",  "Bet (bluff)"),
    "Q":    ("P0  Q  opening",      "Check",  "Bet"),
    "K":    ("P0  K  opening",      "Check",  "Bet (value)"),
    # P1 facing bet
    "J b":  ("P1  J  facing bet",   "Fold",   "Call"),
    "Q b":  ("P1  Q  facing bet",   "Fold",   "Call"),
    "K b":  ("P1  K  facing bet",   "Fold",   "Call"),
    # P1 after check
    "J p":  ("P1  J  after check",  "Check",  "Bet (bluff)"),
    "Q p":  ("P1  Q  after check",  "Check",  "Bet"),
    "K p":  ("P1  K  after check",  "Check",  "Bet (value)"),
    # P0 facing bet after check-bet
    "J pb": ("P0  J  facing raise", "Fold",   "Call"),
    "Q pb": ("P0  Q  facing raise", "Fold",   "Call"),
    "K pb": ("P0  K  facing raise", "Fold",   "Call"),
}

# display order: P0 root, P1 facing bet, P1 after check, P0 facing raise
_DISPLAY_ORDER = [
    "J", "Q", "K",
    "J b", "Q b", "K b",
    "J p", "Q p", "K p",
    "J pb", "Q pb", "K pb",
]


def plot_heatmap(strategy_table):
    """Strategy heatmap: rows = info sets, columns = actions."""
    keys = [k for k in _DISPLAY_ORDER if k in strategy_table]
    labels = []
    for k in keys:
        label, _, _ = _INFO_SET_LABELS.get(k, (k, "pass", "bet"))
        labels.append(label)

    data = np.array([strategy_table[k] for k in keys])
    col_labels = ["Action 0 (pass)", "Action 1 (bet)"]

    fig, ax = plt.subplots(figsize=(7, max(5, len(keys) * 0.42)))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=col_labels, yticklabels=labels,
        vmin=0, vmax=1, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Action Probability"}, ax=ax,
    )
    ax.set_title("Nash Equilibrium Strategy Profile", fontsize=13)
    ax.tick_params(axis="y", rotation=0)

    # draw dividers between player groups (P0 root | P1 bet | P1 check | P0 raise)
    for boundary in [3, 6, 9]:
        if boundary <= len(keys):
            ax.axhline(y=boundary, color="black", linewidth=1.5)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "strategy_heatmap.png"), dpi=150)
    plt.close(fig)
    print("Saved strategy_heatmap.png")


if __name__ == "__main__":
    main()
