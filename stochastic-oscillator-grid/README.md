# Stochastic Oscillator Grid (Frohlich-Style Synchronization)

A Python simulation of a 2D phase lattice where local coupling and stochastic noise compete over time, producing a disorder-to-synchrony transition at a Goldilocks `K/T` ratio.

## Features
- `N x N` NumPy grid of continuous phases `theta in [0, 2π)`.
- Local phase alignment using Kuramoto-style coupling with periodic boundary conditions.
- Configurable neighborhoods:
  - `moore` (8-neighbor, default)
  - `von_neumann` (4-neighbor)
- Gaussian stochastic perturbation with cooling schedule (`exponential` or `linear`).
- Real-time HSV phase visualization using `matplotlib.animation.FuncAnimation`.

## Model Update
At each step:

`theta_i(t+1) = theta_i(t) + dt * (K / deg) * Σ_j sin(theta_j - theta_i) + Normal(0, sqrt(T*dt))`

Then wrap phases back to `[0, 2π)`.

## Defaults
- `N=100`
- `K=0.25`
- `T0=0.8`
- `T_min=0.02`
- `neighborhood=moore`
- periodic (torus-wrap) boundaries

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

Example with custom settings:
```bash
python simulate.py --n 120 --k 0.3 --t0 0.9 --cooling exponential --cooling-rate 0.003 --steps 2500
```

Use 4-neighbor mode:
```bash
python simulate.py --neighborhood von_neumann
```

Save animation:
```bash
python simulate.py --save run.gif
```

## Interpreting Results
- High `T` (low `K/T`): random flickering, weak coherence.
- Intermediate `K/T`: large synchronized domains and propagating waves.
- Very low `T` or weak `K`: over-frozen or fragmented patterns, depending on coupling.
