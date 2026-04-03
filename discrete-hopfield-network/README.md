# Discrete Hopfield Network for Pattern Reconstruction

A standalone Python project that builds a **10x10 discrete Hopfield network** (`N=100`) and demonstrates state-based memory recovery from noisy binary input.

The network is trained on 3 stored patterns:
- `Cross`
- `Square`
- `Diagonal`

Given a corrupted input (default: 30% random bit flips), asynchronous updates drive the system to a low-energy attractor. By default, the script retries with new random corruptions/seeds until **exact reconstruction** is achieved.

## Model

### Hebbian Learning
Weights are trained from stored bipolar patterns (`-1/+1`):

`W = (1/N) * Σ (x x^T) - I`

- `W` is `100 x 100`
- diagonal/self-connections are zeroed

### Energy

`E = -0.5 * Σ_i Σ_j w_ij s_i s_j`

The asynchronous update loop tends to reduce this energy.

### Asynchronous Dynamics

For randomly selected unit `i`:

`s_i = sign(Σ_j w_ij s_j)`

If local field is exactly zero, current state is retained.

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

Default behavior:
- targets `cross`
- applies `30%` corruption
- runs asynchronous relaxation
- retries attempts until an exact match is found
- shows side-by-side evolution: Original → Corrupted → Intermediate → Recovered

## Useful Options

```bash
python simulate.py --pattern cross --flip-rate 0.3 --max-steps 6000 --max-retries 200 --seed 7
```

Save output figure instead of showing a window:

```bash
python simulate.py --save hopfield_result.png
```

Try different stored patterns:

```bash
python simulate.py --pattern square
python simulate.py --pattern diagonal
```

## Success Criterion

The run is successful when a noisy version of `Cross` converges to the **exact clean Cross** with no explicit pattern-recognition `if/else` logic—only attractor dynamics over the learned energy landscape.
