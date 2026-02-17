"""
visualisation.py — Publication-Quality Solver Visualisations
=============================================================

All plot functions render in a dark "technical blueprint" theme
with consistent styling.  Designed for academic figures and
README screenshots.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from typing import Any, Dict, List, Optional

from matplotlib.patches import FancyBboxPatch

from .poker_env import ACTIONS, BoardTexture, HandStrengthModel
from .cfr_solver import CFRSolver


# ── Theme ─────────────────────────────────────────────────────────────

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


def apply_theme() -> None:
    """Apply the dark technical theme globally."""
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


# ── Panel: Convergence ────────────────────────────────────────────────

def plot_convergence(
    regret_history: List[float],
    exploitability_history: List[float],
    ax: plt.Axes,
) -> None:
    """Dual-line convergence plot: mean |regret| and ε upper bound."""
    T = np.arange(1, len(regret_history) + 1)

    ax.plot(T, regret_history, color=ACCENT_BLUE, linewidth=1.3,
            label="Mean |Regret| / T", alpha=0.9)

    if exploitability_history and len(exploitability_history) == len(T):
        ax.plot(T, exploitability_history, color=ACCENT_AMBER,
                linewidth=1.0, linestyle="--", label="ε upper bound / T",
                alpha=0.7)

    ax.axhline(0, color=ACCENT_RED, linewidth=0.6, linestyle=":",
               alpha=0.5, label="Nash equilibrium")
    ax.set_xlabel("Iteration (T)")
    ax.set_ylabel("Regret Metric")
    ax.set_title("Nash Equilibrium Convergence", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )


# ── Panel: Strategy Heatmap ──────────────────────────────────────────

def plot_strategy_heatmap(solver: CFRSolver, ax: plt.Axes) -> None:
    """Heatmap of Hero's average strategy across hand-strength buckets."""
    strategies = solver.get_average_strategies()
    hero_keys = sorted(k for k in strategies if k.startswith("P0|"))

    # Group by HS bucket, average across action histories
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

    im = ax.imshow(
        matrix, aspect="auto", cmap="inferno", vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(ACTIONS)))
    ax.set_yticklabels(ACTIONS, fontsize=8)
    ax.set_xticks(range(n_buckets))
    ax.set_xticklabels(
        [f"{i/n_buckets:.0%}–{(i+1)/n_buckets:.0%}" for i in range(n_buckets)],
        rotation=45, ha="right", fontsize=6,
    )
    ax.set_xlabel("Hand Strength Bucket")
    ax.set_title("Hero Strategy Heatmap", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# ── Panel: Hand Strength Distributions ────────────────────────────────

def plot_hand_distributions(ax: plt.Axes) -> None:
    """Overlay Beta PDFs for all board textures."""
    x = np.linspace(0.005, 0.995, 500)
    colours = {
        BoardTexture.DRY: ACCENT_BLUE,
        BoardTexture.WET: ACCENT_GREEN,
        BoardTexture.NEUTRAL: ACCENT_AMBER,
    }
    for tex in BoardTexture:
        model = HandStrengthModel(board_texture=tex)
        y = model.pdf(x)
        label = f"{tex.name.title()} (α={model.params['alpha']}, β={model.params['beta']})"
        ax.fill_between(x, y, alpha=0.15, color=colours[tex])
        ax.plot(x, y, linewidth=1.4, color=colours[tex], label=label)

    ax.set_xlabel("Hand Strength  s ∈ [0, 1]")
    ax.set_ylabel("Density  f(s)")
    ax.set_title("Beta-Distributed Hand Strength", fontweight="bold")
    ax.legend(fontsize=7, loc="upper center")


# ── Panel: Session Report Box ─────────────────────────────────────────

def plot_report_box(
    solver: CFRSolver,
    exploit_report: Dict[str, Any],
    mdf_threshold: float,
    ax: plt.Axes,
) -> None:
    """Styled text panel with session metrics."""
    ax.axis("off")

    s = solver.summary()
    opp = s["opponent"]
    env = s["env"]
    classification = exploit_report.get("classification", "—")

    # Colour the classification
    cls_colour = ACCENT_RED if classification == "OVER-FOLDING" else ACCENT_GREEN

    lines = [
        ("CFR SOLVER — SESSION REPORT", ACCENT_BLUE, 12),
        ("", None, 6),
        (f"  Iterations         {s['iterations']:>10,d}", TEXT_PRIMARY, 9),
        (f"  Information Sets   {s['info_sets']:>10,d}", TEXT_PRIMARY, 9),
        (f"  Mean |Regret|      {s['final_mean_regret']:>10.4f}", TEXT_PRIMARY, 9),
        (f"  ε Upper Bound      {s['final_exploitability_bound']:>10.4f}", TEXT_PRIMARY, 9),
        ("", None, 4),
        ("  OPPONENT ANALYSIS", ACCENT_AMBER, 10),
        (f"  Fold-to-CBet       {opp['fold_to_cbet']:>9.1%}", TEXT_PRIMARY, 9),
        (f"  MDF Threshold      {mdf_threshold:>9.1%}", TEXT_PRIMARY, 9),
        (f"  Classification     {classification:>12s}", cls_colour, 9),
        ("", None, 4),
    ]

    if env["human_fold_enabled"]:
        lines += [
            ("  HUMAN-FOLD SIGNAL", ACCENT_PURPLE, 10),
            (f"  Signals Fired      {env['human_folds_fired']:>10d}", TEXT_PRIMARY, 9),
            (f"  Adjusted Fold      {opp['adjusted_fold_rate']:>9.1%}", TEXT_PRIMARY, 9),
            ("", None, 4),
        ]

    lines += [
        ("  ENVIRONMENT", TEXT_SECONDARY, 10),
        (f"  Pot / Stack / SPR  {env['pot']} / {env['stack']} / {env['spr']}", TEXT_PRIMARY, 9),
        (f"  Board Texture      {env['board_texture']:>12s}", TEXT_PRIMARY, 9),
    ]

    y = 0.96
    for text, colour, size in lines:
        if colour is None:
            y -= 0.02
            continue
        ax.text(
            0.05, y, text,
            transform=ax.transAxes,
            fontsize=size,
            fontfamily="monospace",
            color=colour,
            verticalalignment="top",
        )
        y -= 0.055

    # Border box
    box = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02",
        facecolor=BG_DARK,
        edgecolor=ACCENT_BLUE,
        linewidth=1.0,
        transform=ax.transAxes,
        zorder=-1,
    )
    ax.add_patch(box)


# ── Panel: Action Distribution Pie ────────────────────────────────────

def plot_opponent_actions(solver: CFRSolver, ax: plt.Axes) -> None:
    """Pie chart of opponent's observed action distribution."""
    opp = solver.opponent_model
    data = [
        opp.fold_count, opp.check_count,
        opp.bet_half_count, opp.bet_pot_count, opp.all_in_count,
    ]
    labels = list(ACTIONS)
    colours = [ACTION_COLOURS[a] for a in ACTIONS]

    nonzero = [(d, l, c) for d, l, c in zip(data, labels, colours) if d > 0]
    if not nonzero:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color=TEXT_SECONDARY, transform=ax.transAxes)
        ax.set_title("Opponent Actions", fontweight="bold")
        return

    vals, labs, cols = zip(*nonzero)
    wedges, texts, autotexts = ax.pie(
        vals, labels=labs, colors=cols, autopct="%1.0f%%",
        startangle=90, textprops={"fontsize": 7, "color": TEXT_PRIMARY},
        pctdistance=0.8, wedgeprops={"edgecolor": BG_DARK, "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(7)
        t.set_color(BG_DARK)
        t.set_fontweight("bold")
    ax.set_title("Opponent Action Distribution", fontweight="bold")


# ── Master Dashboard ──────────────────────────────────────────────────

def render_dashboard(
    solver: CFRSolver,
    exploit_report: Dict[str, Any],
    mdf_threshold: float,
    output_path: str = "cfr_dashboard.png",
    dpi: int = 200,
) -> str:
    """
    Render the full 6-panel dashboard and save to *output_path*.

    Layout (3 × 2):
      [Convergence      ] [Strategy Heatmap  ]
      [Hand Distributions] [Opponent Actions  ]
      [Report Box        ] [  (reserved)      ]
    """
    apply_theme()
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "GTO POKER SOLVER  ·  CFR Nash Equilibrium & Exploitative AI",
        fontsize=14, fontweight="bold", color=ACCENT_BLUE, y=0.985,
    )

    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.28)

    # Row 0
    ax_conv = fig.add_subplot(gs[0, 0])
    plot_convergence(
        solver.regret_history,
        solver.exploitability_history,
        ax_conv,
    )

    ax_heat = fig.add_subplot(gs[0, 1])
    plot_strategy_heatmap(solver, ax_heat)

    # Row 1
    ax_dist = fig.add_subplot(gs[1, 0])
    plot_hand_distributions(ax_dist)

    ax_pie = fig.add_subplot(gs[1, 1])
    plot_opponent_actions(solver, ax_pie)

    # Row 2: full-width report
    ax_report = fig.add_subplot(gs[2, :])
    plot_report_box(solver, exploit_report, mdf_threshold, ax_report)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
