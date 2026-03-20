"""
solve.py -- Run CFR+ on Kuhn Poker and generate result plots
=============================================================

Produces:
    results/convergence.png        -- log-scale exploitability vs iterations
    results/strategy_heatmap.png   -- action frequencies at Nash equilibrium
    results/strategy_matrix.png    -- dust-style Nash strategy profile
    results/convergence_paths.png  -- per-info-set strategy convergence
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from cfr import CFRSolver
from kuhn import CARD_NAMES


def run_solver(iterations=50000):
    solver = CFRSolver(use_cfr_plus=True)
    history = solver.train(iterations)
    gv = solver.game_value()
    eps = history[-1][1]
    print(f"Game value (p0 EV): {gv:.6f}")
    print(f"Final exploitability: {eps:.6f}")
    return solver, history


def _style():
    """Shared style constants for all plots. Poker red/green."""
    return {
        "red": "#d32f2f",
        "green": "#2e7d32",
        "light": "#f5f5f5",
        "mid": "#9e9e9e",
        "font": "DejaVu Sans",
        "title_size": 14,
        "label_size": 12,
        "tick_size": 10,
    }


def plot_convergence(history, out="results/convergence.png"):
    S = _style()
    iters = [x[0] for x in history]
    exploit = [x[1] for x in history]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.semilogy(iters, exploit, color=S["green"], linewidth=1.6, alpha=0.9)
    ax.fill_between(iters, exploit, alpha=0.08, color=S["green"])

    ax.set_xlabel("Iteration", fontsize=S["label_size"], fontfamily=S["font"])
    ax.set_ylabel("Exploitability", fontsize=S["label_size"], fontfamily=S["font"])
    ax.tick_params(labelsize=S["tick_size"])
    ax.grid(True, alpha=0.15, which="both")

    final = exploit[-1]
    ax.annotate(
        f"{final:.4f}", xy=(iters[-1], final),
        xytext=(-70, 30), textcoords="offset points",
        fontsize=S["tick_size"], color=S["red"],
        arrowprops=dict(arrowstyle="->", color=S["red"], lw=1.2),
    )

    ax.set_title(
        "CFR+ Exploitability Convergence\n"
        "Exploitability (log scale) decreasing toward Nash equilibrium",
        fontsize=S["title_size"], fontweight="bold", fontfamily=S["font"], pad=12,
    )

    plt.tight_layout(pad=2.0)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


def plot_strategy_heatmap(solver, out="results/strategy_heatmap.png"):
    S = _style()
    table = solver.get_strategy_table()
    from matplotlib.colors import LinearSegmentedColormap
    cmap_custom = LinearSegmentedColormap.from_list(
        "red_green", [S["red"], S["light"], S["green"]])

    def get_history(k):
        parts = k.split()
        return parts[1] if len(parts) > 1 else ""

    p0_sets = {k: v for k, v in table.items() if len(get_history(k)) % 2 == 0}
    p1_sets = {k: v for k, v in table.items() if len(get_history(k)) % 2 == 1}

    def build(info_sets):
        labels = sorted(info_sets.keys())
        data = np.array([[info_sets[k][0], info_sets[k][1]] for k in labels])
        return labels, data

    p0_labels, p0_data = build(p0_sets)
    p1_labels, p1_data = build(p1_sets)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), facecolor="white",
        gridspec_kw={"wspace": 0.35},
    )

    for ax, labels, data, title in [
        (ax1, p0_labels, p0_data, "Player 1 (first to act)"),
        (ax2, p1_labels, p1_data, "Player 2 (responds)"),
    ]:
        sns.heatmap(
            data, ax=ax, annot=True, fmt=".2f",
            xticklabels=["Check/Fold", "Bet/Call"],
            yticklabels=labels,
            cmap=cmap_custom, vmin=0, vmax=1,
            cbar_kws={"shrink": 0.8},
            linewidths=0.5, linecolor="white",
            annot_kws={"fontsize": S["tick_size"], "fontweight": "bold"},
        )
        ax.set_title(title, fontsize=S["label_size"], fontweight="bold",
                     fontfamily=S["font"], pad=12)
        ax.tick_params(labelsize=S["tick_size"])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=S["tick_size"], fontfamily=S["font"])
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=S["tick_size"],
                           fontfamily=S["font"])

    fig.suptitle(
        "Nash Equilibrium Strategy Profile\n"
        "Kuhn Poker  |  CFR+ 50k Iterations  |  Action Frequencies",
        fontsize=S["title_size"], fontweight="bold", fontfamily=S["font"],
        y=1.02,
    )

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.88)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


def plot_strategy_dust(solver, out="results/strategy_matrix.png"):
    """
    Dust-style 12x2 strategy matrix. Each cell filled with 500 scatter
    points colored by strategy probability. Deep Crimson / Steel Blue.
    """
    S = _style()
    table = solver.get_strategy_table()
    rng = np.random.default_rng(42)
    pts = 500

    def get_history(k):
        parts = k.split()
        return parts[1] if len(parts) > 1 else ""

    p0_keys = sorted([k for k in table if len(get_history(k)) % 2 == 0])
    p1_keys = sorted([k for k in table if len(get_history(k)) % 2 == 1])
    all_keys = p0_keys + p1_keys
    nrows = len(all_keys)

    fig, axes = plt.subplots(nrows, 2, figsize=(8, 18), facecolor="white")

    for i, key in enumerate(all_keys):
        strat = table[key]
        probs = [strat[0], strat[1]]
        for j in range(2):
            ax = axes[i, j]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])

            prob = probs[j]
            n_colored = int(prob * pts)
            n_gray = pts - n_colored
            color = S["red"] if j == 0 else S["green"]

            if n_colored > 0:
                px = rng.uniform(0.02, 0.98, size=n_colored)
                py = rng.uniform(0.02, 0.98, size=n_colored)
                ax.scatter(px, py, s=1.5, alpha=0.6, color=color,
                           rasterized=True)
            if n_gray > 0:
                px = rng.uniform(0.02, 0.98, size=n_gray)
                py = rng.uniform(0.02, 0.98, size=n_gray)
                ax.scatter(px, py, s=1.5, alpha=0.08, color=S["mid"],
                           rasterized=True)

            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')
                spine.set_linewidth(0.3)

            if i == 0:
                col_label = "Check / Fold" if j == 0 else "Bet / Call"
                ax.set_title(col_label, fontsize=S["label_size"],
                             fontweight="bold", fontfamily=S["font"], pad=8)

            if j == 0:
                ax.set_ylabel(key, fontsize=S["tick_size"],
                              fontfamily=S["font"], rotation=0,
                              labelpad=35, va="center")

    # Divider between P1 and P2 sections
    for j in range(2):
        axes[len(p0_keys) - 1, j].spines["bottom"].set_linewidth(2.0)
        axes[len(p0_keys) - 1, j].spines["bottom"].set_edgecolor(S["mid"])

    fig.suptitle(
        "Nash Equilibrium Strategy Profile (Dust Map)\n"
        "12 Information Sets x 2 Actions  |  CFR+ 50k Iterations",
        fontsize=S["title_size"], fontweight="bold", fontfamily=S["font"],
        y=0.995,
    )

    legend_elements = [
        Patch(facecolor=S["red"], label="Check/Fold"),
        Patch(facecolor=S["green"], label="Bet/Call"),
        Patch(facecolor=S["mid"], alpha=0.3, label="Complement"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=S["tick_size"], framealpha=0.95,
               prop={"family": S["font"]},
               bbox_to_anchor=(0.5, 0.002))

    plt.tight_layout(pad=2.5, h_pad=0.8)
    plt.subplots_adjust(top=0.94, bottom=0.04)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


def plot_convergence_paths(solver, out="results/convergence_paths.png"):
    """
    Per-info-set strategy convergence paths.
    12 lines using a crimson-to-steel gradient.
    """
    S = _style()
    hist = solver.strategy_history
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "rg", [S["red"], S["mid"], S["green"]])
    sorted_keys = sorted(hist.keys())
    n = len(sorted_keys)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")

    for idx, key in enumerate(sorted_keys):
        data = hist[key]
        iters = [d[0] for d in data]
        bets = [d[1] for d in data]
        ax.plot(iters, bets, linewidth=0.8, alpha=0.7, color=colors[idx])

    for idx, key in enumerate(sorted_keys):
        data = hist[key]
        final_iter = data[-1][0]
        final_val = data[-1][1]
        ax.annotate(key, xy=(final_iter, final_val),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=8, color=colors[idx], fontfamily=S["font"],
                    va="center")

    ax.set_xlabel("Iteration", fontsize=S["label_size"], fontfamily=S["font"])
    ax.set_ylabel("Bet Probability", fontsize=S["label_size"], fontfamily=S["font"])
    ax.tick_params(labelsize=S["tick_size"])
    ax.set_title(
        "Strategy Convergence Paths: All 12 Information Sets\n"
        "CFR+ 50k Iterations  |  Bet probability settling toward Nash equilibrium",
        fontsize=S["title_size"], fontweight="bold", fontfamily=S["font"], pad=12,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.3, alpha=0.3)
    ax.axhline(1, color="black", linewidth=0.3, alpha=0.3)

    plt.tight_layout(pad=2.0)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


def print_strategy_table(solver):
    table = solver.get_strategy_table()
    print("\n=== Nash Equilibrium Strategy ===\n")
    print(f"{'Info Set':<12} {'Check/Fold':>12} {'Bet/Call':>12}")
    print("-" * 38)
    for key, strat in table.items():
        print(f"{key:<12} {strat[0]:>12.3f} {strat[1]:>12.3f}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print("Running CFR+ on Kuhn Poker (50000 iterations)...\n")

    solver, history = run_solver(iterations=50000)
    print_strategy_table(solver)
    print()

    plot_convergence(history)
    plot_strategy_heatmap(solver)
    plot_strategy_dust(solver)
    plot_convergence_paths(solver)
    print("\ndone.")
