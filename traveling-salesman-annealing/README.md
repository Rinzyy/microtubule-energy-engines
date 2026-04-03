# Traveling Salesman Optimizer via Simulated Annealing

A standalone Python simulation that uses stochastic, energy-based search to optimize a route through random cities.

## Objective
- State: ordered route (permutation of city indices)
- Energy: total closed-loop route distance
- Goal: minimize route distance using a Goldilocks temperature schedule

## Core Update Rule
At each annealing step:
1. Propose a mutation by swapping two cities in the route.
2. Compute `ΔE = E_new - E_current`.
3. If `ΔE < 0`, always accept.
4. If `ΔE > 0`, accept with probability:

`P = exp(-ΔE / T)`

This thermal acceptance allows occasional uphill moves that help escape local traps.

## Features
- Default `20` random cities (`--cities` configurable).
- Geometric annealing schedule with temperature floor.
- Tracks current and best-so-far route energy.
- Real-time route animation (current route + best route overlay).
- Optional output save:
  - `.gif` / `.mp4`: save full animation
  - other extension (e.g., `.png`): save final frame

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python simulate.py
```

Useful options:

```bash
python simulate.py --cities 20 --steps 12000 --t0 1.5 --cooling-rate 0.9993 --seed 7
```

Save output:

```bash
python simulate.py --save tsp_run.gif
python simulate.py --save tsp_final.png
```

## Success Criteria
A successful run should show:
- a significantly shorter final best route than the random starting route,
- visible route untangling from chaotic crossings to cleaner loops,
- nonzero uphill accepts (thermal kicks), demonstrating local-minima escape.
