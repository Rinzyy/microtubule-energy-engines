# Physarum Polycephalum Simulation

A 2D agent-based slime mold simulation where thousands of simple particles move, sense, and deposit trails to form emergent transport networks.

## Model Summary

- **Trail Map**: 2D NumPy array (`grid_size x grid_size`) storing scent concentration.
- **Agents**: vectorized arrays of position `(x, y)` and heading `phi`.
- **Boundary**: torus wrap-around for both motion and sensing.

Each simulation step has three phases:
1. **Motor stage**: agents move forward and deposit scent.
2. **Sensory stage**: each agent samples forward/left/right sensors and turns toward strongest scent.
3. **Environment stage**: trail diffuses (local neighborhood mixing via `np.roll`) and decays (evaporation factor).

No SciPy is used; diffusion is implemented with NumPy-only local interactions.

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

## Key Options

```bash
python simulate.py \
  --grid-size 512 \
  --agents 10000 \
  --steps 6000 \
  --sensor-distance 9 \
  --sensor-angle-deg 45 \
  --turn-angle-deg 25 \
  --deposit 1.0 \
  --diffuse 0.22 \
  --decay 0.96
```

- `--save output.mp4` or `--save output.gif` saves animation.
- `--save output.png` saves final frame.
- If performance is limited, reduce `--grid-size` before reducing `--agents`.

## Expected Behavior

The initial random field should self-organize into branching veins and highway-like channels. The structure remains dynamic because decay prevents permanent saturation, forcing continual re-optimization of trails.
