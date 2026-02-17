"""
main.py — Entry Point for the GTO Poker Solver
================================================

Pipeline:
  1. Configure the river endgame environment.
  2. Run CFR (or CFR+) training.
  3. Evaluate exploitative adjustments via MDF criterion.
  4. Generate a publication-quality dashboard.

Usage:
  python -m gto_poker_solver.main
  python -m gto_poker_solver.main --iterations 50000 --human-fold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gto_poker_solver.poker_env import BoardTexture, RiverEndgameEnv
from gto_poker_solver.cfr_solver import CFRSolver
from gto_poker_solver.visualisation import render_dashboard


# ── Defaults ──────────────────────────────────────────────────────────

DEFAULTS = dict(
    iterations=10_000,
    pot=100.0,
    stack=200.0,
    board="DRY",
    mdf=0.50,
    seed=42,
    human_fold=False,
    output="cfr_dashboard.png",
    json_report=None,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gto-poker-solver",
        description="CFR Nash Equilibrium Solver for NLHE River Endgames",
    )
    p.add_argument("-n", "--iterations", type=int,
                   default=DEFAULTS["iterations"],
                   help="Number of CFR iterations (default: %(default)s)")
    p.add_argument("--pot", type=float, default=DEFAULTS["pot"])
    p.add_argument("--stack", type=float, default=DEFAULTS["stack"])
    p.add_argument("--board", choices=["DRY", "WET", "NEUTRAL"],
                   default=DEFAULTS["board"])
    p.add_argument("--mdf", type=float, default=DEFAULTS["mdf"],
                   help="MDF fold-frequency threshold (default: %(default)s)")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--human-fold", action="store_true",
                   help="Enable human-fold-as-signal tactic")
    p.add_argument("-o", "--output", default=DEFAULTS["output"],
                   help="Dashboard output path (default: %(default)s)")
    p.add_argument("--json-report", default=DEFAULTS["json_report"],
                   help="Optional JSON report output path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    texture = BoardTexture[args.board]

    # ── Banner ────────────────────────────────────────────────────
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║       GTO POKER SOLVER  v2.1                        ║")
    print("  ║       CFR Nash Equilibrium & Exploitative AI        ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    # ── 1. Environment ────────────────────────────────────────────
    env = RiverEndgameEnv(
        pot=args.pot,
        stack=args.stack,
        board_texture=texture,
        enable_human_fold=args.human_fold,
        seed=args.seed,
    )
    print(f"  Environment: pot={args.pot}  stack={args.stack}  "
          f"SPR={env.spr:.1f}  board={texture.name}")
    if args.human_fold:
        print("  Human-Fold-as-Signal: ENABLED")
    print()

    # ── 2. CFR Training ──────────────────────────────────────────
    print("  ┌─ Phase 1: CFR+ Training ──────────────────────────┐")
    solver = CFRSolver(env, use_cfr_plus=True)
    solver.train(iterations=args.iterations)
    print("  └──────────────────────────────────────────────────────┘")
    print()

    solver.print_summary()
    print()

    # ── 3. Exploitative Adjustment ────────────────────────────────
    print("  ┌─ Phase 2: Exploitative Adjustment ────────────────┐")
    exploit_report = solver.exploit_weakness(mdf_threshold=args.mdf)
    print("  └──────────────────────────────────────────────────────┘")
    print()

    # ── 4. Visualisation ─────────────────────────────────────────
    print("  ┌─ Phase 3: Dashboard Rendering ────────────────────┐")
    path = render_dashboard(
        solver, exploit_report, args.mdf, output_path=args.output,
    )
    print(f"  Saved → {path}")
    print("  └──────────────────────────────────────────────────────┘")
    print()

    # ── 5. Optional JSON report ───────────────────────────────────
    if args.json_report:
        report = {**solver.summary(), "exploit": exploit_report}
        Path(args.json_report).write_text(
            json.dumps(report, indent=2, default=str)
        )
        print(f"  JSON report → {args.json_report}")

    print("  Done.\n")


if __name__ == "__main__":
    main()
