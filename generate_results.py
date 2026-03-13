"""
generate_results.py
Runs the CFR solver and produces all figures for the README.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from scipy.stats import beta as beta_dist

from gto_poker_solver.poker_env import (
    ACTIONS, BoardTexture, HandStrengthModel, RiverEndgameEnv
)
from gto_poker_solver.cfr_solver import CFRSolver

# ── Theme ──────────────────────────────────────────────────────────
BG_DARK = "#0a0e14"
BG_PANEL = "#111820"
BORDER = "#1e2a38"
GRID = "#172030"
TEXT_PRIMARY = "#e0e8f0"
TEXT_SECONDARY = "#6b7d94"
ACCENT_BLUE = "#4da6ff"
ACCENT_GREEN = "#3ddc84"
ACCENT_RED = "#ff5252"
ACCENT_AMBER = "#ffab40"
ACCENT_PURPLE = "#bb86fc"

ACTION_COLOURS = {
    "FOLD": ACCENT_RED,
    "CHECK": TEXT_SECONDARY,
    "CALL": ACCENT_BLUE,
    "BET_HALF": ACCENT_GREEN,
    "BET_POT": ACCENT_AMBER,
    "ALL_IN": ACCENT_PURPLE,
}


def set_theme():
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": BG_PANEL,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.facecolor": BG_PANEL,
        "legend.edgecolor": BORDER,
        "font.family": "monospace",
        "font.size": 9,
    })


def run_solver():
    """Train CFR on dry board, 10k iterations."""
    print("Training CFR+ solver (10,000 iterations)...")
    env = RiverEndgameEnv(
        pot=100.0, stack=200.0,
        board_texture=BoardTexture.DRY,
        enable_human_fold=True,
        seed=42,
    )
    solver = CFRSolver(env, use_cfr_plus=True)
    solver.train(iterations=10_000, log_every=2500)
    report = solver.exploit_weakness(mdf_threshold=0.50)
    return solver, report


def fig_convergence(solver, out="results/convergence.png"):
    """Plot convergence of mean regret and exploitability bound."""
    set_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    T = np.arange(1, len(solver.regret_history) + 1)

    ax.plot(T, solver.regret_history, color=ACCENT_BLUE, linewidth=1.3,
            label="Mean |Regret| / T", alpha=0.9)
    ax.plot(T, solver.exploitability_history, color=ACCENT_AMBER,
            linewidth=1.0, linestyle="--", label="Exploitability upper bound",
            alpha=0.7)
    ax.axhline(0, color=ACCENT_RED, linewidth=0.6, linestyle=":",
               alpha=0.5, label="Nash equilibrium (zero regret)")

    ax.set_xlabel("Iteration (T)")
    ax.set_ylabel("Regret Metric")
    ax.set_title("CFR+ Convergence to Nash Equilibrium", fontweight="bold", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_strategy_heatmap(solver, out="results/strategy_heatmap.png"):
    """Heatmap: Hero's average strategy across hand-strength buckets."""
    set_theme()
    strategies = solver.get_average_strategies()
    hero_keys = sorted(k for k in strategies if k.startswith("P0|"))

    n_buckets = solver.env.hand_buckets
    matrix = np.zeros((len(ACTIONS), n_buckets))
    counts = np.zeros(n_buckets)

    for key in hero_keys:
        try:
            bucket = int(key.split("|")[1].replace("HS", ""))
        except (IndexError, ValueError):
            continue
        if 0 <= bucket < n_buckets:
            matrix[:, bucket] += strategies[key]
            counts[bucket] += 1

    for b in range(n_buckets):
        if counts[b] > 0:
            matrix[:, b] /= counts[b]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="inferno", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(ACTIONS)))
    ax.set_yticklabels(ACTIONS, fontsize=9)
    ax.set_xticks(range(n_buckets))
    ax.set_xticklabels(
        [f"{i/n_buckets:.0%}-{(i+1)/n_buckets:.0%}" for i in range(n_buckets)],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_xlabel("Hand Strength Bucket")
    ax.set_title("Hero's Equilibrium Strategy by Hand Strength", fontweight="bold", fontsize=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Action Probability", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_beta_distributions(out="results/beta_distributions.png"):
    """Overlay Beta PDFs for dry/wet/neutral board textures."""
    set_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0.005, 0.995, 500)
    colours = {
        BoardTexture.DRY: ACCENT_BLUE,
        BoardTexture.WET: ACCENT_GREEN,
        BoardTexture.NEUTRAL: ACCENT_AMBER,
    }
    for tex in BoardTexture:
        model = HandStrengthModel(board_texture=tex)
        y = model.pdf(x)
        label = f"{tex.name.title()} board (a={model.params['alpha']}, b={model.params['beta']})"
        ax.fill_between(x, y, alpha=0.15, color=colours[tex])
        ax.plot(x, y, linewidth=1.4, color=colours[tex], label=label)

    ax.set_xlabel("Hand Strength s in [0, 1]")
    ax.set_ylabel("Density f(s)")
    ax.set_title("Hand Strength Distributions by Board Texture", fontweight="bold", fontsize=12)
    ax.legend(fontsize=8, loc="upper center")

    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_full_dashboard(solver, report, out="results/dashboard.png"):
    """Full 6-panel dashboard."""
    set_theme()
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "GTO POKER SOLVER  |  CFR Nash Equilibrium Dashboard",
        fontsize=14, fontweight="bold", color=ACCENT_BLUE, y=0.985,
    )
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.28)

    # Row 0 left: convergence
    ax = fig.add_subplot(gs[0, 0])
    T = np.arange(1, len(solver.regret_history) + 1)
    ax.plot(T, solver.regret_history, color=ACCENT_BLUE, linewidth=1.3, label="Mean |Regret| / T")
    ax.plot(T, solver.exploitability_history, color=ACCENT_AMBER, linewidth=1.0, linestyle="--", label="Exploitability bound")
    ax.axhline(0, color=ACCENT_RED, linewidth=0.6, linestyle=":", alpha=0.5)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Regret")
    ax.set_title("Convergence", fontweight="bold"); ax.legend(fontsize=7)

    # Row 0 right: heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    strategies = solver.get_average_strategies()
    hero_keys = sorted(k for k in strategies if k.startswith("P0|"))
    n_b = solver.env.hand_buckets
    mat = np.zeros((len(ACTIONS), n_b)); cnts = np.zeros(n_b)
    for key in hero_keys:
        try:
            b = int(key.split("|")[1].replace("HS", ""))
        except: continue
        if 0 <= b < n_b: mat[:, b] += strategies[key]; cnts[b] += 1
    for b in range(n_b):
        if cnts[b] > 0: mat[:, b] /= cnts[b]
    im = ax2.imshow(mat, aspect="auto", cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax2.set_yticks(range(len(ACTIONS))); ax2.set_yticklabels(ACTIONS, fontsize=8)
    ax2.set_xticks(range(n_b))
    ax2.set_xticklabels([f"{i/n_b:.0%}" for i in range(n_b)], fontsize=7)
    ax2.set_xlabel("Hand Strength"); ax2.set_title("Strategy Heatmap", fontweight="bold")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # Row 1 left: beta distributions
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.linspace(0.005, 0.995, 500)
    colours_map = {BoardTexture.DRY: ACCENT_BLUE, BoardTexture.WET: ACCENT_GREEN, BoardTexture.NEUTRAL: ACCENT_AMBER}
    for tex in BoardTexture:
        model = HandStrengthModel(board_texture=tex)
        y = model.pdf(x)
        ax3.fill_between(x, y, alpha=0.15, color=colours_map[tex])
        ax3.plot(x, y, linewidth=1.4, color=colours_map[tex], label=f"{tex.name.title()}")
    ax3.set_xlabel("Hand Strength"); ax3.set_ylabel("Density")
    ax3.set_title("Beta Distributions", fontweight="bold"); ax3.legend(fontsize=7)

    # Row 1 right: opponent action pie
    ax4 = fig.add_subplot(gs[1, 1])
    opp = solver.opponent_model
    data = [opp.fold_count, opp.check_count, opp.call_count,
            opp.bet_half_count, opp.bet_pot_count, opp.all_in_count]
    labels = list(ACTIONS)
    cols = [ACTION_COLOURS[a] for a in ACTIONS]
    nonzero = [(d, l, c) for d, l, c in zip(data, labels, cols) if d > 0]
    if nonzero:
        vals, labs, cs = zip(*nonzero)
        ax4.pie(vals, labels=labs, colors=cs, autopct="%1.0f%%", startangle=90,
                textprops={"fontsize": 7, "color": TEXT_PRIMARY}, pctdistance=0.8,
                wedgeprops={"edgecolor": BG_DARK, "linewidth": 1.5})
    ax4.set_title("Opponent Actions", fontweight="bold")

    # Row 2: report panel
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")
    s = solver.summary()
    opp_d = s["opponent"]; env_d = s["env"]
    cls = report.get("classification", "?")
    cls_c = ACCENT_RED if cls == "OVER-FOLDING" else ACCENT_GREEN
    lines = [
        ("CFR SOLVER SESSION REPORT", ACCENT_BLUE, 12),
        ("", None, 6),
        (f"  Iterations         {s['iterations']:>10,d}", TEXT_PRIMARY, 9),
        (f"  Information Sets   {s['info_sets']:>10,d}", TEXT_PRIMARY, 9),
        (f"  Mean |Regret|      {s['final_mean_regret']:>10.4f}", TEXT_PRIMARY, 9),
        (f"  Exploitability     {s['final_exploitability_bound']:>10.4f}", TEXT_PRIMARY, 9),
        ("", None, 4),
        ("  OPPONENT ANALYSIS", ACCENT_AMBER, 10),
        (f"  Fold-to-CBet       {opp_d['fold_to_cbet']:>9.1%}", TEXT_PRIMARY, 9),
        (f"  Classification     {cls:>12s}", cls_c, 9),
        ("", None, 4),
        ("  ENVIRONMENT", TEXT_SECONDARY, 10),
        (f"  Pot / Stack / SPR  {env_d['pot']} / {env_d['stack']} / {env_d['spr']}", TEXT_PRIMARY, 9),
        (f"  Board Texture      {env_d['board_texture']:>12s}", TEXT_PRIMARY, 9),
    ]
    if env_d["human_fold_enabled"]:
        lines.insert(-2, ("  HUMAN-FOLD SIGNAL", ACCENT_PURPLE, 10))
        lines.insert(-2, (f"  Signals Fired      {env_d['human_folds_fired']:>10d}", TEXT_PRIMARY, 9))
        lines.insert(-2, (f"  Adjusted Fold      {opp_d['adjusted_fold_rate']:>9.1%}", TEXT_PRIMARY, 9))

    y_pos = 0.96
    for text, colour, size in lines:
        if colour is None: y_pos -= 0.02; continue
        ax5.text(0.05, y_pos, text, transform=ax5.transAxes, fontsize=size,
                 fontfamily="monospace", color=colour, verticalalignment="top")
        y_pos -= 0.055
    box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
                         facecolor=BG_DARK, edgecolor=ACCENT_BLUE, linewidth=1.0,
                         transform=ax5.transAxes, zorder=-1)
    ax5.add_patch(box)

    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    solver, report = run_solver()
    print("\nGenerating figures...")
    fig_convergence(solver)
    fig_strategy_heatmap(solver)
    fig_beta_distributions()
    fig_full_dashboard(solver, report)
    print("\nDone. All figures saved to results/")
