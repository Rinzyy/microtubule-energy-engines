# Microtubule Simulation Playground

This repository contains four standalone simulation projects designed to build intuition around state-based dynamics, energy minimization, and stochastic search ideas connected to microtubule-inspired and quantum-cellular-automaton-style thinking.

## Projects

### 1) `stochastic-oscillator-grid`
A 2D noisy phase-lattice simulation (Frohlich-style synchronization) showing how local coupling and noise can produce global order at a Goldilocks `K/T` ratio.

- Core idea: disorder-to-coherence transition
- Stack: `numpy`, `matplotlib`
- Run: `python simulate.py`

### 2) `discrete-hopfield-network`
A 10x10 Hopfield attractor network that stores binary patterns and relaxes noisy inputs into learned memories through asynchronous updates.

- Core idea: memory as energy-landscape descent
- Stack: `numpy`, `matplotlib`
- Run: `python simulate.py`

### 3) `traveling-salesman-annealing`
A Traveling Salesman optimizer using simulated annealing where thermal acceptance (`exp(-ΔE/T)`) helps escape local minima and untangle routes.

- Core idea: stochastic optimization via annealing
- Stack: `numpy`, `matplotlib`
- Run: `python simulate.py`

### 4) `physarum-polycephalum-simulation`
A vectorized slime mold simulation where thousands of simple agents move, sense trails, and self-organize into vein-like transport networks.

- Core idea: emergent intelligence in a shared state lattice
- Stack: `numpy`, `matplotlib`
- Run: `python simulate.py`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r stochastic-oscillator-grid/requirements.txt
pip install -r discrete-hopfield-network/requirements.txt
pip install -r traveling-salesman-annealing/requirements.txt
pip install -r physarum-polycephalum-simulation/requirements.txt
```

Then run any project from its folder.

## Notes

- Each project is intentionally standalone (`simulate.py` + `README.md` + `requirements.txt`).
- Generated outputs like `.png`, `.gif`, and `.mp4` are ignored by default in `.gitignore`.
