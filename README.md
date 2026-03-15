<p align="center">
  <img src="results/poker_banner.jpg" width="720" />
</p>

# CFR+ Nash Equilibrium Solver for Imperfect Information Games

A from-scratch implementation of **Counterfactual Regret Minimization (CFR+)** applied to Kuhn Poker, a canonical benchmark in algorithmic game theory. The solver computes Nash equilibrium strategies through iterative self-play and regret matching, converging to game-theoretic optimal play in a provably correct manner.

## What the project finds

The solver recovers the analytical Nash equilibrium of Kuhn Poker after ~10,000 CFR+ iterations:

| Info Set | Pass/Check/Fold | Bet/Call | Interpretation |
|----------|-----------------|----------|----------------|
| P0: J (root) | 0.749 | 0.251 | Bluff ~1/4 of the time |
| P0: Q (root) | 0.999 | 0.001 | Never open with Q |
| P0: K (root) | 0.250 | 0.750 | Value bet ~3/4 |
| P1: J facing bet | 1.000 | 0.000 | Always fold |
| P1: Q facing bet | 0.662 | 0.338 | Call ~1/3 |
| P1: K facing bet | 0.000 | 1.000 | Always call |

The computed game value converges to **-1/18 = -0.0556**, matching the known analytical result from Kuhn (1950). Exploitability drops below 0.001 by 50,000 iterations, confirming convergence to a Nash equilibrium.

### Convergence

<p align="center">
  <img src="results/convergence.png" width="600" />
</p>

### Strategy Profile

<p align="center">
  <img src="results/strategy_heatmap.png" width="540" />
</p>

## How it works

**1. Game definition (`kuhn.py`)**
Kuhn Poker uses three cards (J, Q, K) dealt to two players. Each player antes 1 chip, then alternates between pass (check/fold) and bet (bet/call). The game tree has 5 terminal histories: pp, bp, bb, pbp, pbb.

**2. CFR+ engine (`cfr.py`)**
The solver traverses the full game tree for all 6 possible card deals each iteration. At every information set, it performs regret matching to derive the current strategy, then updates cumulative regrets. CFR+ floors negative regrets to zero (Tammelin 2014), accelerating convergence. The average strategy across all iterations converges to a Nash equilibrium.

**3. Convergence verification**
Exploitability is computed via information-set-level best response: for each player, enumerate all pure strategies over their info sets and find the one maximizing (or minimizing) the expected value. At Nash, no player can unilaterally improve, so exploitability equals zero.

## Project structure

```
gto-poker-solver/
    kuhn.py              # game rules, terminal payoffs, info set keys
    cfr.py               # CFR+ solver, regret matching, exploitability
    solve.py             # training loop, plot generation
    requirements.txt     # numpy, matplotlib, seaborn
    results/
        convergence.png       # log-scale exploitability vs iterations
        strategy_heatmap.png  # Nash strategy profile per info set
        poker_banner.jpg      # header image
```

## Running

```bash
pip install -r requirements.txt
python solve.py
```

Results are saved to `results/`.

## References

1. Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). "Regret Minimization in Games with Incomplete Information." *Advances in Neural Information Processing Systems (NeurIPS)*.

2. Tammelin, O. (2014). "Solving Large Imperfect Information Games Using CFR+." *arXiv:1407.5042*.

3. Kuhn, H. W. (1950). "Simplified Two-Person Poker." *Contributions to the Theory of Games*, 1, 97-103.

## License

MIT License. See [LICENSE](LICENSE).
