import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


GRID_SIZE = 10
N_UNITS = GRID_SIZE * GRID_SIZE


@dataclass
class Config:
    flip_rate: float = 0.3
    max_steps: int = 6000
    max_retries: int = 200
    seed: int | None = 7
    pattern: str = "cross"
    save: str | None = None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Discrete Hopfield Network for 10x10 pattern reconstruction"
    )
    parser.add_argument(
        "--flip-rate",
        type=float,
        default=0.3,
        help="Fraction of bits to flip for corruption (default: 0.3)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6000,
        help="Asynchronous updates per attempt (default: 6000)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=200,
        help="Max corruption/recovery attempts until exact match (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base RNG seed; each retry uses seed+attempt (default: 7)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        choices=["cross", "square", "diagonal"],
        default="cross",
        help="Stored memory to corrupt/recover (default: cross)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save figure (e.g., result.png)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.flip_rate <= 1.0:
        raise ValueError("--flip-rate must be in [0, 1]")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be > 0")

    return Config(
        flip_rate=args.flip_rate,
        max_steps=args.max_steps,
        max_retries=args.max_retries,
        seed=args.seed,
        pattern=args.pattern,
        save=args.save,
    )


def make_patterns() -> dict[str, np.ndarray]:
    cross = -np.ones((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    cross[GRID_SIZE // 2, :] = 1
    cross[:, GRID_SIZE // 2] = 1

    square = -np.ones((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    square[1:-1, 1] = 1
    square[1:-1, -2] = 1
    square[1, 1:-1] = 1
    square[-2, 1:-1] = 1

    diagonal = -np.ones((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    idx = np.arange(GRID_SIZE)
    diagonal[idx, idx] = 1

    return {"cross": cross, "square": square, "diagonal": diagonal}


def flatten_pattern(pattern_2d: np.ndarray) -> np.ndarray:
    return pattern_2d.reshape(-1).astype(np.float64)


def reshape_state(state_flat: np.ndarray) -> np.ndarray:
    return state_flat.reshape(GRID_SIZE, GRID_SIZE)


def train_hebbian(patterns: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(patterns, axis=0)
    w = (stacked.T @ stacked) / N_UNITS
    w -= np.eye(N_UNITS)
    np.fill_diagonal(w, 0.0)
    return w


def energy(w: np.ndarray, state: np.ndarray) -> float:
    return float(-0.5 * state @ w @ state)


def corrupt_pattern(clean: np.ndarray, flip_rate: float, rng: np.random.Generator) -> np.ndarray:
    corrupted = clean.copy()
    n_flip = int(round(flip_rate * N_UNITS))
    if n_flip > 0:
        flip_indices = rng.choice(N_UNITS, size=n_flip, replace=False)
        corrupted[flip_indices] *= -1
    return corrupted


def sign_update(value: float, current: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return current


def run_async_relaxation(
    w: np.ndarray,
    initial: np.ndarray,
    target: np.ndarray,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray], list[float], bool, int]:
    state = initial.copy()
    snapshots: list[np.ndarray] = [state.copy()]
    energies: list[float] = [energy(w, state)]

    for step in range(1, max_steps + 1):
        i = int(rng.integers(0, N_UNITS))
        local_field = float(w[i] @ state)
        state[i] = sign_update(local_field, state[i])

        if step % 100 == 0 or np.array_equal(state, target):
            snapshots.append(state.copy())
            energies.append(energy(w, state))

        if np.array_equal(state, target):
            return state, snapshots, energies, True, step

    return state, snapshots, energies, False, max_steps


def pick_intermediate_snapshots(snapshots: list[np.ndarray], count: int = 4) -> list[np.ndarray]:
    if len(snapshots) <= 2:
        return []
    if len(snapshots) <= count + 2:
        return snapshots[1:-1]

    idx = np.linspace(1, len(snapshots) - 2, num=count, dtype=int)
    return [snapshots[i] for i in idx]


def render_evolution(
    original: np.ndarray,
    corrupted: np.ndarray,
    intermediate: list[np.ndarray],
    recovered: np.ndarray,
    energies: list[float],
    success: bool,
    steps: int,
    attempt: int,
    pattern_name: str,
    save_path: str | None,
) -> None:
    panels = [original, corrupted, *intermediate, recovered]
    titles = ["Original", "Corrupted", *[f"Step {i + 1}" for i in range(len(intermediate))], "Recovered"]

    ncols = len(panels)
    fig, axes = plt.subplots(2, ncols, figsize=(2.4 * ncols, 6.2), gridspec_kw={"height_ratios": [4, 1]})

    for idx, (panel, title) in enumerate(zip(panels, titles)):
        ax = axes[0, idx]
        ax.imshow(reshape_state(panel), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(ncols):
        axes[1, idx].axis("off")

    energy_ax = axes[1, 0]
    energy_ax.axis("on")
    energy_ax.plot(energies, color="tab:red", linewidth=1.8)
    energy_ax.set_title("Energy during asynchronous updates")
    energy_ax.set_xlabel("Recorded snapshot index")
    energy_ax.set_ylabel("E")
    energy_ax.grid(alpha=0.25)

    status = "SUCCESS" if success else "FAILED"
    fig.suptitle(
        f"Hopfield Pattern Reconstruction ({pattern_name}) | {status} | attempts={attempt} | steps={steps}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def main() -> None:
    cfg = parse_args()
    patterns_2d = make_patterns()
    stored = {name: flatten_pattern(pattern) for name, pattern in patterns_2d.items()}

    w = train_hebbian(list(stored.values()))
    target = stored[cfg.pattern]

    successful_result = None

    for attempt in range(1, cfg.max_retries + 1):
        if cfg.seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(cfg.seed + attempt)

        corrupted = corrupt_pattern(target, cfg.flip_rate, rng)
        final_state, snapshots, energies, success, steps = run_async_relaxation(
            w=w,
            initial=corrupted,
            target=target,
            max_steps=cfg.max_steps,
            rng=rng,
        )

        if success:
            successful_result = (corrupted, final_state, snapshots, energies, steps, attempt)
            break

    if successful_result is None:
        raise RuntimeError(
            "No exact recovery found within max retries. "
            "Try increasing --max-retries or --max-steps, or lowering --flip-rate."
        )

    corrupted, final_state, snapshots, energies, steps, attempt = successful_result
    intermediate = pick_intermediate_snapshots(snapshots, count=4)

    print(
        f"Exact recovery achieved for '{cfg.pattern}' in attempt {attempt}, "
        f"steps={steps}, final_energy={energy(w, final_state):.4f}"
    )

    render_evolution(
        original=target,
        corrupted=corrupted,
        intermediate=intermediate,
        recovered=final_state,
        energies=energies,
        success=True,
        steps=steps,
        attempt=attempt,
        pattern_name=cfg.pattern,
        save_path=cfg.save,
    )


if __name__ == "__main__":
    main()
