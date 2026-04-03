import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


@dataclass
class Config:
    cities: int = 20
    steps: int = 12000
    t0: float = 1.5
    t_min: float = 1e-4
    cooling_rate: float = 0.9993
    interval_ms: int = 25
    updates_per_frame: int = 50
    seed: int | None = 7
    save: str | None = None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Traveling Salesman optimization via simulated annealing"
    )
    parser.add_argument("--cities", type=int, default=20, help="Number of random cities")
    parser.add_argument("--steps", type=int, default=12000, help="Total annealing updates")
    parser.add_argument("--t0", type=float, default=1.5, help="Initial temperature")
    parser.add_argument("--t-min", type=float, default=1e-4, help="Minimum temperature floor")
    parser.add_argument(
        "--cooling-rate",
        type=float,
        default=0.9993,
        help="Geometric cooling multiplier per update (0,1)",
    )
    parser.add_argument(
        "--interval-ms", type=int, default=25, help="Animation frame interval in ms"
    )
    parser.add_argument(
        "--updates-per-frame",
        type=int,
        default=50,
        help="How many annealing updates are batched into one animation frame",
    )
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path (.gif/.mp4 for animation, other extension saves final frame)",
    )
    args = parser.parse_args()

    if args.cities < 5:
        raise ValueError("--cities must be >= 5")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.t0 <= 0:
        raise ValueError("--t0 must be > 0")
    if args.t_min <= 0:
        raise ValueError("--t-min must be > 0")
    if not 0.0 < args.cooling_rate < 1.0:
        raise ValueError("--cooling-rate must be in (0,1)")
    if args.interval_ms <= 0:
        raise ValueError("--interval-ms must be > 0")
    if args.updates_per_frame <= 0:
        raise ValueError("--updates-per-frame must be > 0")

    return Config(
        cities=args.cities,
        steps=args.steps,
        t0=args.t0,
        t_min=args.t_min,
        cooling_rate=args.cooling_rate,
        interval_ms=args.interval_ms,
        updates_per_frame=args.updates_per_frame,
        seed=args.seed,
        save=args.save,
    )


def route_energy(cities: np.ndarray, route: np.ndarray) -> float:
    ordered = cities[route]
    shifted = np.roll(ordered, shift=-1, axis=0)
    deltas = shifted - ordered
    return float(np.sum(np.sqrt(np.sum(deltas * deltas, axis=1))))


def mutate_swap(route: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    i, j = rng.choice(route.size, size=2, replace=False)
    candidate = route.copy()
    candidate[i], candidate[j] = candidate[j], candidate[i]
    return candidate


def temperature(step: int, cfg: Config) -> float:
    return max(cfg.t_min, cfg.t0 * (cfg.cooling_rate**step))


def closed_route_points(cities: np.ndarray, route: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = cities[route]
    pts = np.vstack([pts, pts[0]])
    return pts[:, 0], pts[:, 1]


def run_annealing(cfg: Config) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    cities = rng.random((cfg.cities, 2))

    current_route = np.arange(cfg.cities)
    rng.shuffle(current_route)
    current_energy = route_energy(cities, current_route)

    best_route = current_route.copy()
    best_energy = current_energy
    initial_energy = current_energy

    accepted_moves = 0
    accepted_uphill = 0

    history_routes = [current_route.copy()]
    history_best_routes = [best_route.copy()]
    history_steps = [0]
    history_temp = [cfg.t0]
    history_current_energy = [current_energy]
    history_best_energy = [best_energy]

    for step in range(1, cfg.steps + 1):
        t = temperature(step, cfg)
        candidate = mutate_swap(current_route, rng)
        candidate_energy = route_energy(cities, candidate)
        delta = candidate_energy - current_energy

        accept = False
        if delta <= 0.0:
            accept = True
        else:
            if rng.random() < np.exp(-delta / max(t, 1e-12)):
                accept = True
                accepted_uphill += 1

        if accept:
            current_route = candidate
            current_energy = candidate_energy
            accepted_moves += 1
            if current_energy < best_energy:
                best_energy = current_energy
                best_route = current_route.copy()

        if step % cfg.updates_per_frame == 0 or step == cfg.steps:
            history_routes.append(current_route.copy())
            history_best_routes.append(best_route.copy())
            history_steps.append(step)
            history_temp.append(t)
            history_current_energy.append(current_energy)
            history_best_energy.append(best_energy)

    return {
        "cities": cities,
        "initial_energy": initial_energy,
        "final_current_energy": current_energy,
        "best_energy": best_energy,
        "best_route": best_route,
        "history_routes": history_routes,
        "history_best_routes": history_best_routes,
        "history_steps": history_steps,
        "history_temp": history_temp,
        "history_current_energy": history_current_energy,
        "history_best_energy": history_best_energy,
        "accepted_moves": accepted_moves,
        "accepted_uphill": accepted_uphill,
        "total_moves": cfg.steps,
    }


def make_animation(cfg: Config, result: dict[str, object]) -> tuple[plt.Figure, FuncAnimation]:
    cities = result["cities"]
    history_routes = result["history_routes"]
    history_best_routes = result["history_best_routes"]
    history_steps = result["history_steps"]
    history_temp = result["history_temp"]
    history_current_energy = result["history_current_energy"]
    history_best_energy = result["history_best_energy"]

    assert isinstance(cities, np.ndarray)
    assert isinstance(history_routes, list)
    assert isinstance(history_best_routes, list)
    assert isinstance(history_steps, list)
    assert isinstance(history_temp, list)
    assert isinstance(history_current_energy, list)
    assert isinstance(history_best_energy, list)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.scatter(cities[:, 0], cities[:, 1], s=28, color="black", zorder=3)

    line_current, = ax.plot([], [], color="tab:blue", linewidth=1.5, alpha=0.7, label="Current route")
    line_best, = ax.plot([], [], color="tab:orange", linewidth=2.4, alpha=0.95, label="Best-so-far")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    improvement = 100.0 * (result["initial_energy"] - result["best_energy"]) / max(result["initial_energy"], 1e-12)

    def update(frame: int):
        route = history_routes[frame]
        best_route = history_best_routes[frame]

        x_curr, y_curr = closed_route_points(cities, route)
        x_best, y_best = closed_route_points(cities, best_route)

        line_current.set_data(x_curr, y_curr)
        line_best.set_data(x_best, y_best)

        ax.set_title(
            (
                f"TSP Simulated Annealing | step={history_steps[frame]} | "
                f"T={history_temp[frame]:.5f}\n"
                f"current={history_current_energy[frame]:.4f} | "
                f"best={history_best_energy[frame]:.4f} | "
                f"improvement={improvement:.2f}%"
            )
        )
        return [line_current, line_best]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history_routes),
        interval=cfg.interval_ms,
        blit=True,
        repeat=False,
    )

    return fig, animation


def main() -> None:
    cfg = parse_args()
    result = run_annealing(cfg)

    initial = float(result["initial_energy"])
    final_current = float(result["final_current_energy"])
    best = float(result["best_energy"])
    improvement = 100.0 * (initial - best) / max(initial, 1e-12)
    accepted = int(result["accepted_moves"])
    uphill = int(result["accepted_uphill"])
    total = int(result["total_moves"])

    print(f"Initial distance: {initial:.4f}")
    print(f"Final current distance: {final_current:.4f}")
    print(f"Best distance found: {best:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Accepted moves: {accepted}/{total} ({100.0 * accepted / total:.2f}%)")
    print(f"Accepted uphill moves: {uphill}")

    fig, animation = make_animation(cfg, result)

    if cfg.save:
        lower = cfg.save.lower()
        if lower.endswith(".gif") or lower.endswith(".mp4"):
            animation.save(cfg.save, dpi=120)
            print(f"Saved animation to: {cfg.save}")
        else:
            fig.savefig(cfg.save, dpi=150)
            print(f"Saved final frame to: {cfg.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
