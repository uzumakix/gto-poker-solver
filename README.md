# ♠ GTO Poker Solver

**CFR Nash Equilibrium & Exploitative AI for No-Limit Hold'em River Endgames**

<br>

<div align="center">

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![CI](https://img.shields.io/github/actions/workflow/status/your-org/gto-poker-solver/ci.yml?style=flat-square&label=CI)
![Game Theory](https://img.shields.io/badge/domain-game%20theory-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/status-research--grade-orange?style=flat-square)

</div>

<br>

> A research-grade implementation of **Counterfactual Regret Minimization (CFR+)** that computes Nash equilibrium strategies, detects exploitable opponents via the Minimum Defence Frequency criterion, and introduces a novel **human-fold-as-signal** layer for meta-game deception against automated tracking systems.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Counterfactual Regret Minimization](#3-counterfactual-regret-minimization)
4. [Regret Matching](#4-regret-matching)
5. [Exploitative Criterion — Minimum Defence Frequency](#5-exploitative-criterion--minimum-defence-frequency)
6. [Beta-Distributed Hand Strength Model](#6-beta-distributed-hand-strength-model)
7. [Human-Fold-as-Signal — The Meta-Game Layer](#7-human-fold-as-signal--the-meta-game-layer)
8. [Zero-Sum Terminal Payoff](#8-zero-sum-terminal-payoff)
9. [Installation & Usage](#9-installation--usage)
10. [Dashboard Output](#10-dashboard-output)
11. [Testing](#11-testing)
12. [References](#12-references)

---

## 1. Overview

This solver addresses the **river endgame** — the final betting round in No-Limit Hold'em — as a two-player zero-sum extensive-form game with imperfect information.  The two players are referred to as **Hero** (player 0) and **Opponent** (player 1) throughout.

The system provides three layers of strategic computation:

| Layer | Method | Purpose |
|:---|:---|:---|
| **Equilibrium** | CFR+ self-play | Compute approximate Nash equilibrium strategies |
| **Exploitation** | MDF-based deviation | Maximise EV against opponents who over-fold |
| **Meta-game** | Human-fold-as-signal | Corrupt opposing bots' tracking models via deliberate folds |

All terminology adheres to standard game-theoretic conventions.  No colloquialisms are used.

---

## 2. Architecture

```
gto-poker-solver/
│
├── .github/workflows/ci.yml             # GitHub Actions CI (lint + test matrix)
├── pyproject.toml                       # PEP 621 packaging & tool config
├── requirements.txt                     # Dependency pins (pip fallback)
├── README.md                            # This document
├── LICENSE                              # MIT
│
├── gto_poker_solver/
│   ├── __init__.py                      # Public API + version
│   ├── py.typed                         # PEP 561 type marker
│   ├── poker_env.py                     # Environment: states, actions, payoffs
│   ├── cfr_solver.py                    # CFR+ engine, opponent model, exploitation
│   ├── visualisation.py                 # 6-panel dark-theme dashboard renderer
│   └── main.py                          # CLI entry point (argparse)
│
├── tests/
│   ├── conftest.py                      # Shared fixtures
│   ├── test_env.py                      # Environment + payoff tests
│   └── test_solver.py                   # CFR, regret matching, exploit tests
│
└── assets/
    ├── dashboard_preview.png            # README screenshot
    └── sample_report.json               # Example JSON output
```

**Module dependency graph:**

```
main.py ─────┬──▶ cfr_solver.py ──▶ poker_env.py
             │
             └──▶ visualisation.py ──┬──▶ cfr_solver.py
                                     └──▶ poker_env.py
```

---

## 3. Counterfactual Regret Minimization

### 3.1 Extensive-Form Games and Information Sets

A finite two-player zero-sum game in extensive form is defined by a game tree in which each non-terminal node belongs to exactly one player.  In imperfect-information settings, nodes are partitioned into **information sets** $\mathcal{I}_i$ for each player $i$; a player cannot distinguish between nodes within the same information set.

### 3.2 Counterfactual Value

For player $i$, the **counterfactual value** of an information set $I$ under strategy profile $\sigma$ is:

$$v_i^\sigma(I) = \sum_{h \in I}\; \pi_{-i}^\sigma(h) \sum_{z \in Z(h)} \pi^\sigma(h, z)\, u_i(z)$$

where $\pi_{-i}^\sigma(h)$ is the reach probability of history $h$ due to all players *except* $i$, $Z(h)$ is the set of terminal histories reachable from $h$, and $u_i(z)$ is the terminal payoff for player $i$ at leaf $z$.

### 3.3 Regret and Convergence

The **instantaneous counterfactual regret** for action $a$ at information set $I$ on iteration $t$ is:

$$r^t(I, a) = v_i^{\sigma^t}(I \!\to\! a) - v_i^{\sigma^t}(I)$$

Cumulative counterfactual regret after $T$ iterations:

$$R^T(I, a) = \sum_{t=1}^{T} r^t(I, a)$$

The **exploitability** $\epsilon$ of the average strategy profile satisfies:

$$\epsilon \;\leq\; \frac{1}{T}\,\sum_{I \in \mathcal{I}_i} \max_a R^T(I, a)^{+} \;\xrightarrow{T \to \infty}\; 0$$

### 3.4 CFR+ Enhancement

This implementation uses **CFR+** (Tammelin, 2014), which floors negative cumulative regrets to zero after each iteration:

$$R^{t,+}(I, a) = \max\!\Big(R^{t-1,+}(I, a) + r^t(I, a),\; 0\Big)$$

CFR+ converges faster in practice because it prevents old negative regrets from delaying the emergence of strategies that later prove beneficial.

---

## 4. Regret Matching

At each iteration the current strategy is derived from cumulative regrets:

$$\sigma^{t+1}(I, a) = \begin{cases}
\dfrac{R^t(I, a)^{+}}{\displaystyle\sum_{b}\, R^t(I, b)^{+}} & \text{if } \displaystyle\sum_{b} R^t(I, b)^{+} > 0 \\[12pt]
\dfrac{1}{|A(I)|} & \text{otherwise}
\end{cases}$$

where $x^{+} = \max(x, 0)$ and $A(I)$ is the set of legal actions.

The **time-averaged strategy** is the one that converges to Nash equilibrium:

$$\bar{\sigma}(I, a) = \frac{\displaystyle\sum_{t=1}^{T}\, \pi_i^t(I)\, \sigma^t(I, a)}{\displaystyle\sum_{t=1}^{T}\, \pi_i^t(I)}$$

---

## 5. Exploitative Criterion — Minimum Defence Frequency

### 5.1 Definition

The **Minimum Defence Frequency** (MDF) specifies the fraction of a player's range that must continue (call or raise) against a bet to deny the bettor automatic profit.  For a bet of size $b$ into pot $p$:

$$\text{MDF} = \frac{p}{p + b}$$

A player who folds more often than $1 - \text{MDF}$ is **over-folding** and is vulnerable to exploitative aggression.

### 5.2 Detection & Adjustment

The solver tracks the Opponent's empirical fold frequency $\hat{f}_{\text{opp}}$ across all training iterations.  When:

$$\hat{f}_{\text{opp}} > \text{threshold}$$

the solver applies a **maximally exploitative strategy**: for information sets where Hero holds the bottom 20% of the range (hands with negligible showdown equity), the strategy is overridden to a pure All-In.

The expected value of this bluff is:

$$\text{EV}_{\text{bluff}} = \hat{f}_{\text{opp}} \cdot p - (1 - \hat{f}_{\text{opp}}) \cdot b > 0 \quad \Longleftrightarrow \quad \hat{f}_{\text{opp}} > \frac{b}{p + b}$$

This is not a heuristic — it is the **best response** in a two-player zero-sum game.

---

## 6. Beta-Distributed Hand Strength Model

### 6.1 Motivation

Rather than enumerating the full combinatorial card space, hand strength is abstracted to a scalar $s \in [0, 1]$ drawn from a **Beta distribution** conditioned on board texture.

### 6.2 Parameterisation

$$s \sim \text{Beta}(\alpha, \beta)$$

| Board Texture | $\alpha$ | $\beta$ | Shape | Strategic Interpretation |
|:---:|:---:|:---:|:---|:---|
| **Dry** | 0.80 | 0.80 | U-shaped (bimodal) | Polarised ranges — hands cluster at extremes |
| **Wet** | 2.50 | 2.50 | Bell-shaped | Merged ranges — hands cluster near the median |
| **Neutral** | 1.20 | 1.20 | Mildly uniform | Moderate polarisation |

The probability density function:

$$f(s;\, \alpha, \beta) = \frac{s^{\alpha - 1}\,(1 - s)^{\beta - 1}}{B(\alpha, \beta)} \qquad\text{where}\quad B(\alpha, \beta) = \frac{\Gamma(\alpha)\,\Gamma(\beta)}{\Gamma(\alpha + \beta)}$$

---

## 7. Human-Fold-as-Signal — The Meta-Game Layer

### 7.1 The Problem with Bot Tracking Systems

Automated opponents (bots) build **opponent models** by recording action frequencies — particularly fold-to-bet ratios.  These models assume that every fold is a genuine concession of equity.  This assumption creates a vulnerability.

### 7.2 The Deliberate Fold

A human player (or a bot emulating human behaviour) can **intentionally fold a strong hand** in a spot where folding appears natural.  From the opposing bot's perspective, this fold is indistinguishable from a standard weak-hand concession.

The effect is a **poisoned observation**: the opposing bot's tracker records an additional fold, inflating its estimate of the player's fold-to-bet frequency.  On subsequent hands, the bot will assign higher probability to the player folding and will therefore increase its bluff frequency — at which point the player traps with strong holdings.

### 7.3 Formal Model

The signal is deployed when three conditions hold simultaneously:

| Condition | Parameter | Default | Meaning |
|:---|:---|:---:|:---|
| Hand strength ≥ floor | `hand_strength_floor` | 0.65 | Only sacrifice hands with real equity |
| Sacrifice ≤ cap | `sacrifice_ev_cap` | 0.35 | Limit EV loss on the sacrificed hand |
| Random activation | `activation_probability` | 0.15 | Stochastic to avoid detection patterns |

The **cost** of the signal is the EV forfeited on the current hand.  The **benefit** is accumulated across subsequent hands where the corrupted opponent model leads the bot to over-bluff.  The tactic is profitable when:

$$\sum_{t=\tau+1}^{T} \Delta\text{EV}_t^{\text{trap}} > \text{EV}_\tau^{\text{sacrificed}}$$

where $\tau$ is the signal hand and $\Delta\text{EV}_t^{\text{trap}}$ is the incremental value gained from the bot's over-bluffing on hand $t$.

### 7.4 Why This Works Against Bots

Bots that use simple frequency trackers (fold counts, continuation-bet statistics) treat every action at face value.  They lack a **theory of mind** — the capacity to reason about whether an opponent's action was *intentional* rather than *informational*.  The deliberate fold exploits exactly this gap.

A bot with a more sophisticated model (e.g., one that tracks timing tells, bet-sizing patterns, or models intentional deception) would be more resistant.  The solver's `OpponentModel` class includes an `adjusted_fold_rate` property that subtracts known human-fold signals, demonstrating how such a defence might work.

### 7.5 Enabling the Feature

```bash
python -m gto_poker_solver.main --human-fold
```

The dashboard will display the number of signals fired and the adjusted fold rate.

---

## 8. Zero-Sum Terminal Payoff

The game is **strictly zero-sum** at every terminal node.  For a pot of size $P$:

| Outcome | Hero Payoff | Opponent Payoff | Sum |
|:---|:---:|:---:|:---:|
| Opponent folds | $+P/2$ | $-P/2$ | $0$ |
| Hero folds | $-P/2$ | $+P/2$ | $0$ |
| Showdown — Hero wins | $+P/2$ | $-P/2$ | $0$ |
| Showdown — Opponent wins | $-P/2$ | $+P/2$ | $0$ |
| Showdown — Tie | $0$ | $0$ | $0$ |

The payoff $P/2$ represents the *net gain above each player's own contribution*.  This is critical: returning the full pot on a fold would violate the zero-sum invariant and break CFR's convergence guarantees.

---

## 9. Installation & Usage

### Quick Start

```bash
git clone https://github.com/your-org/gto-poker-solver.git
cd gto-poker-solver
pip install -r requirements.txt
python -m gto_poker_solver.main
```

### Full CLI Options

```bash
python -m gto_poker_solver.main \
    --iterations 50000 \
    --pot 150 \
    --stack 300 \
    --board WET \
    --mdf 0.55 \
    --human-fold \
    --output results.png \
    --json-report session.json
```

| Flag | Default | Description |
|:---|:---:|:---|
| `-n`, `--iterations` | 10,000 | Number of CFR self-play iterations |
| `--pot` | 100 | Starting pot size |
| `--stack` | 200 | Effective stack |
| `--board` | DRY | Board texture: `DRY`, `WET`, `NEUTRAL` |
| `--mdf` | 0.50 | MDF fold-frequency threshold |
| `--seed` | 42 | RNG seed for reproducibility |
| `--human-fold` | off | Enable human-fold-as-signal tactic |
| `-o`, `--output` | `cfr_dashboard.png` | Dashboard image output path |
| `--json-report` | none | Optional JSON report output path |

---

## 10. Dashboard Output

The solver generates a publication-quality 6-panel dashboard (dark technical theme):

<p align="center">
  <img src="assets/dashboard_preview.png" width="820" alt="CFR Solver Dashboard" />
</p>

| Panel | Content |
|:---|:---|
| **Convergence** | Mean \|Regret\|/T and ε upper bound — both converging toward zero |
| **Strategy Heatmap** | Hero's averaged strategy across all 10 hand-strength buckets |
| **Hand Distributions** | Beta PDFs for Dry, Wet, and Neutral board textures |
| **Opponent Actions** | Pie chart of the Opponent's empirical action distribution |
| **Session Report** | Solver metrics, opponent classification, human-fold statistics |

---

## 11. Testing

```bash
# With pytest installed:
python -m pytest tests/ -v

# Or standalone:
python tests/test_solver.py
```

Test coverage includes:

- **Zero-sum invariant** — payoff sums to zero at every terminal node.
- **Fold payoff = ±pot/2** — not the full pot.
- **Beta distribution** — dry boards are polarised; wet boards are merged.
- **Regret matching** — uniform under zero regret; proportional under positive regret.
- **Human-fold signal** — respects strength floor and sacrifice cap.
- **Exploit criterion** — reads `opponent_model.fold_to_cbet`, not Hero's fold rate.

---

## 12. References

1. Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). *Regret Minimization in Games with Incomplete Information*. Advances in Neural Information Processing Systems (NeurIPS).

2. Tammelin, O. (2014). *Solving Large Imperfect Information Games Using CFR+*. arXiv:1407.5042.

3. Brown, N. & Sandholm, T. (2019). *Superhuman AI for Multiplayer Poker*. Science, 365(6456), 885–890.

4. Neller, T. W. & Lanctot, M. (2013). *An Introduction to Counterfactual Regret Minimization*. Teaching material.

5. Hart, S. & Mas-Colell, A. (2000). *A Simple Adaptive Procedure Leading to Correlated Equilibrium*. Econometrica, 68(5), 1127–1150.

---

## License

MIT — released for academic and research purposes.
