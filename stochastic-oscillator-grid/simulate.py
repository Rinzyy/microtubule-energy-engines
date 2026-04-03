import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


TAU = 2.0 * np.pi


@dataclass
class Config:
    n: int = 100
    k: float = 0.25
    t0: float = 0.8
    t_min: float = 0.02
    cooling_rate: float = 0.002
    cooling: str = "exponential"
    neighborhood: str = "moore"
    dt: float = 0.05
    interval_ms: int = 30
    steps: int = 2000
    seed: int | None = None
    save: str | None = None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Stochastic oscillator grid (Frohlich-style synchronization)"
    )
    parser.add_argument("--n", type=int, default=100, help="Grid size (N x N)")
    parser.add_argument("--k", type=float, default=0.25, help="Coupling strength")
    parser.add_argument("--t0", type=float, default=0.8, help="Initial noise intensity")
    parser.add_argument("--t-min", type=float, default=0.02, help="Minimum noise floor")
    parser.add_argument(
        "--cooling-rate",
        type=float,
        default=0.002,
        help="Cooling rate for temperature schedule",
    )
    parser.add_argument(
        "--cooling",
        choices=["exponential", "linear"],
        default="exponential",
        help="Temperature schedule",
    )
    parser.add_argument(
        "--neighborhood",
        choices=["moore", "von_neumann"],
        default="moore",
        help="Local interaction neighborhood",
    )
    parser.add_argument("--dt", type=float, default=0.05, help="Integration step size")
    parser.add_argument(
        "--interval-ms", type=int, default=30, help="Animation frame interval"
    )
    parser.add_argument("--steps", type=int, default=2000, help="Total simulation steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path to save animation (.gif or .mp4)",
    )
    args = parser.parse_args()
    return Config(
        n=args.n,
        k=args.k,
        t0=args.t0,
        t_min=args.t_min,
        cooling_rate=args.cooling_rate,
        cooling=args.cooling,
        neighborhood=args.neighborhood,
        dt=args.dt,
        interval_ms=args.interval_ms,
        steps=args.steps,
        seed=args.seed,
        save=args.save,
    )


def temperature(step: int, cfg: Config) -> float:
    if cfg.cooling == "linear":
        t = cfg.t0 - cfg.cooling_rate * step
    else:
        t = cfg.t0 * np.exp(-cfg.cooling_rate * step)
    return max(cfg.t_min, t)


def neighbor_offsets(mode: str) -> list[tuple[int, int]]:
    if mode == "von_neumann":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]


def coupling_term(theta: np.ndarray, offsets: list[tuple[int, int]]) -> np.ndarray:
    total = np.zeros_like(theta)
    for di, dj in offsets:
        neighbor = np.roll(np.roll(theta, shift=di, axis=0), shift=dj, axis=1)
        total += np.sin(neighbor - theta)
    return total


def run(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed)
    theta = rng.uniform(0.0, TAU, size=(cfg.n, cfg.n))
    offsets = neighbor_offsets(cfg.neighborhood)
    degree = float(len(offsets))

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    image = ax.imshow(theta, cmap="hsv", vmin=0.0, vmax=TAU, animated=True)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Phase θ")
    ax.set_xticks([])
    ax.set_yticks([])

    def step(frame: int):
        nonlocal theta
        t = temperature(frame, cfg)
        coupling = coupling_term(theta, offsets)
        noise = rng.normal(loc=0.0, scale=np.sqrt(t * cfg.dt), size=theta.shape)
        theta = (theta + cfg.dt * (cfg.k / degree) * coupling + noise) % TAU

        image.set_data(theta)
        ratio = cfg.k / max(t, 1e-12)
        ax.set_title(
            (
                f"Stochastic Oscillator Grid | step={frame} | "
                f"T={t:.4f} | K={cfg.k:.3f} | K/T={ratio:.3f}"
            )
        )
        return [image]

    animation = FuncAnimation(
        fig,
        step,
        frames=cfg.steps,
        interval=cfg.interval_ms,
        blit=True,
        repeat=False,
    )

    if cfg.save:
        animation.save(cfg.save, dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    run(parse_args())
