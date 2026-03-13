"""
generate_results.py
Runs the solver and produces figures for the repo.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import beta as beta_dist

from gto_poker_solver.poker_env import (
    ACTIONS, BoardTexture, HandStrengthModel, RiverEndgameEnv
)
from gto_poker_solver.cfr_solver import CFRSolver


# ── Custom style (not the default matplotlib look) ─────────────────
def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#fafafa",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#cccccc",
        "axes.labelcolor": "#333333",
        "axes.grid": True,
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.5,
        "text.color": "#333333",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def run_solver(n=10000):
    print(f"Running CFR+ for {n} iterations...")
    env = RiverEndgameEnv(
        pot=100.0, stack=200.0,
        board_texture=BoardTexture.DRY,
        enable_human_fold=True,
        seed=42,
    )
    solver = CFRSolver(env, use_cfr_plus=True)
    solver.train(iterations=n, log_every=n // 4)
    report = solver.exploit_weakness(mdf_threshold=0.50)
    return solver, report


def fig_convergence(solver, out="results/convergence.png"):
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    T = np.arange(1, len(solver.regret_history) + 1)

    # left: regret
    ax1.plot(T, solver.regret_history, color="#2563eb", linewidth=1.2)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("mean |regret| / T")
    ax1.set_title("cumulative regret (normalized)")
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))

    # right: exploitability
    ax2.plot(T, solver.exploitability_history, color="#dc2626", linewidth=1.2)
    ax2.axhline(0, color="#999999", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("exploitability upper bound")
    ax2.set_title("exploitability (should go to 0)")
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))

    fig.tight_layout(pad=2.0)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_strategy(solver, out="results/strategy_heatmap.png"):
    setup_style()
    strategies = solver.get_average_strategies()
    hero_keys = sorted(k for k in strategies if k.startswith("P0|"))

    n_b = solver.env.hand_buckets
    matrix = np.zeros((len(ACTIONS), n_b))
    counts = np.zeros(n_b)

    for key in hero_keys:
        try:
            b = int(key.split("|")[1].replace("HS", ""))
        except (IndexError, ValueError):
            continue
        if 0 <= b < n_b:
            matrix[:, b] += strategies[key]
            counts[b] += 1

    for b in range(n_b):
        if counts[b] > 0:
            matrix[:, b] /= counts[b]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_yticks(range(len(ACTIONS)))
    ax.set_yticklabels(ACTIONS, fontsize=9)
    ax.set_xticks(range(n_b))
    ax.set_xticklabels([f"HS{i}" for i in range(n_b)], fontsize=8)
    ax.set_xlabel("hand strength bucket (0=weakest, 9=strongest)")
    ax.set_title("hero's equilibrium strategy across hand strengths")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("probability", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_beta(out="results/hand_distributions.png"):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    x = np.linspace(0.01, 0.99, 300)

    configs = [
        (BoardTexture.DRY, "#2563eb", "dry board (a=0.8, b=0.8)\npolarized: nuts or air"),
        (BoardTexture.WET, "#16a34a", "wet board (a=2.5, b=2.5)\nmerged ranges"),
        (BoardTexture.NEUTRAL, "#d97706", "neutral (a=1.2, b=1.2)\nmoderate spread"),
    ]

    for ax, (tex, color, title) in zip(axes, configs):
        model = HandStrengthModel(board_texture=tex)
        y = model.pdf(x)
        ax.fill_between(x, y, alpha=0.25, color=color)
        ax.plot(x, y, linewidth=1.5, color=color)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("hand strength s")
        ax.set_xlim(0, 1)

    axes[0].set_ylabel("density f(s)")
    fig.suptitle("Beta-distributed hand strength by board texture", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_action_bars(solver, out="results/opponent_actions.png"):
    setup_style()
    opp = solver.opponent_model
    data = {
        "fold": opp.fold_count,
        "check": opp.check_count,
        "call": opp.call_count,
        "bet 1/2": opp.bet_half_count,
        "bet pot": opp.bet_pot_count,
        "all-in": opp.all_in_count,
    }
    # filter zeros
    data = {k: v for k, v in data.items() if v > 0}
    total = sum(data.values())

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#ef4444", "#6b7280", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6"]
    bars = ax.barh(list(data.keys()), list(data.values()), color=colors[:len(data)])

    for bar, v in zip(bars, data.values()):
        pct = v / total * 100
        ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9, color="#555555")

    ax.set_xlabel("count")
    ax.set_title(f"opponent action distribution (n={total:,})")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    solver, report = run_solver()
    print()
    fig_convergence(solver)
    fig_strategy(solver)
    fig_beta()
    fig_action_bars(solver)

    # print some stats for the README
    s = solver.summary()
    print(f"\n--- stats for README ---")
    print(f"iterations: {s['iterations']}")
    print(f"info sets: {s['info_sets']}")
    print(f"final exploitability: {s['final_exploitability_bound']:.4f}")
    print(f"opp fold rate: {s['opponent']['fold_to_cbet']:.1%}")
    print(f"human fold signals: {s['opponent']['human_fold_signals']}")
    print("done.")
