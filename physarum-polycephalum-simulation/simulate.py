import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


@dataclass
class Config:
    grid_size: int = 512
    agents: int = 10000
    steps: int = 6000
    speed: float = 1.0
    sensor_distance: float = 9.0
    sensor_angle_deg: float = 45.0
    turn_angle_deg: float = 25.0
    deposit: float = 1.0
    decay: float = 0.96
    diffuse: float = 0.22
    interval_ms: int = 20
    updates_per_frame: int = 10
    seed: int | None = 7
    cmap: str = "inferno"
    save: str | None = None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Physarum polycephalum style agent-trail simulation"
    )
    parser.add_argument("--grid-size", type=int, default=512, help="Trail map size")
    parser.add_argument("--agents", type=int, default=10000, help="Number of agents")
    parser.add_argument("--steps", type=int, default=6000, help="Total simulation steps")
    parser.add_argument("--speed", type=float, default=1.0, help="Agent movement speed")
    parser.add_argument(
        "--sensor-distance", type=float, default=9.0, help="Distance to sensor probes"
    )
    parser.add_argument(
        "--sensor-angle-deg", type=float, default=45.0, help="Left/right sensor angle"
    )
    parser.add_argument(
        "--turn-angle-deg", type=float, default=25.0, help="Heading change per step"
    )
    parser.add_argument("--deposit", type=float, default=1.0, help="Trail deposit per agent")
    parser.add_argument(
        "--decay", type=float, default=0.96, help="Trail evaporation factor per step"
    )
    parser.add_argument(
        "--diffuse", type=float, default=0.22, help="Local diffusion blend in [0, 1]"
    )
    parser.add_argument(
        "--interval-ms", type=int, default=20, help="Animation frame interval"
    )
    parser.add_argument(
        "--updates-per-frame",
        type=int,
        default=10,
        help="Simulation updates per animation frame",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--cmap",
        type=str,
        default="inferno",
        choices=["inferno", "magma", "plasma", "viridis"],
        help="Colormap for trail rendering",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path (.gif/.mp4 for animation, other extension for final frame)",
    )
    args = parser.parse_args()

    if args.grid_size < 64:
        raise ValueError("--grid-size must be >= 64")
    if args.agents < 100:
        raise ValueError("--agents must be >= 100")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.speed <= 0:
        raise ValueError("--speed must be > 0")
    if args.sensor_distance <= 0:
        raise ValueError("--sensor-distance must be > 0")
    if not 0.0 < args.decay <= 1.0:
        raise ValueError("--decay must be in (0, 1]")
    if not 0.0 <= args.diffuse <= 1.0:
        raise ValueError("--diffuse must be in [0, 1]")
    if args.interval_ms <= 0:
        raise ValueError("--interval-ms must be > 0")
    if args.updates_per_frame <= 0:
        raise ValueError("--updates-per-frame must be > 0")

    return Config(
        grid_size=args.grid_size,
        agents=args.agents,
        steps=args.steps,
        speed=args.speed,
        sensor_distance=args.sensor_distance,
        sensor_angle_deg=args.sensor_angle_deg,
        turn_angle_deg=args.turn_angle_deg,
        deposit=args.deposit,
        decay=args.decay,
        diffuse=args.diffuse,
        interval_ms=args.interval_ms,
        updates_per_frame=args.updates_per_frame,
        seed=args.seed,
        cmap=args.cmap,
        save=args.save,
    )


class PhysarumSimulation:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.size = cfg.grid_size
        self.rng = np.random.default_rng(cfg.seed)

        self.x = self.rng.uniform(0.0, self.size, size=cfg.agents).astype(np.float32)
        self.y = self.rng.uniform(0.0, self.size, size=cfg.agents).astype(np.float32)
        self.phi = self.rng.uniform(0.0, 2.0 * np.pi, size=cfg.agents).astype(np.float32)
        self.trail = np.zeros((self.size, self.size), dtype=np.float32)

        self.sensor_angle = np.float32(np.deg2rad(cfg.sensor_angle_deg))
        self.turn_angle = np.float32(np.deg2rad(cfg.turn_angle_deg))

        self.step_count = 0

    def _sense(self, angle_offset: float) -> np.ndarray:
        a = self.phi + angle_offset
        sx = (self.x + self.cfg.sensor_distance * np.cos(a)) % self.size
        sy = (self.y + self.cfg.sensor_distance * np.sin(a)) % self.size

        ix = np.floor(sx).astype(np.int32) % self.size
        iy = np.floor(sy).astype(np.int32) % self.size
        return self.trail[iy, ix]

    def _deposit(self) -> None:
        ix = np.floor(self.x).astype(np.int32) % self.size
        iy = np.floor(self.y).astype(np.int32) % self.size
        np.add.at(self.trail, (iy, ix), self.cfg.deposit)

    def _diffuse_and_decay(self) -> None:
        t = self.trail
        neighbor_sum = (
            np.roll(t, 1, axis=0)
            + np.roll(t, -1, axis=0)
            + np.roll(t, 1, axis=1)
            + np.roll(t, -1, axis=1)
            + np.roll(np.roll(t, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(t, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(t, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(t, -1, axis=0), -1, axis=1)
        )
        mean_3x3 = (t + neighbor_sum) / 9.0
        self.trail = ((1.0 - self.cfg.diffuse) * t + self.cfg.diffuse * mean_3x3) * self.cfg.decay

    def step(self) -> None:
        self.x = (self.x + self.cfg.speed * np.cos(self.phi)) % self.size
        self.y = (self.y + self.cfg.speed * np.sin(self.phi)) % self.size

        self._deposit()

        forward = self._sense(0.0)
        left = self._sense(self.sensor_angle)
        right = self._sense(-self.sensor_angle)

        turn_left = (left > forward) & (left > right)
        turn_right = (right > forward) & (right > left)

        self.phi = self.phi + self.turn_angle * turn_left.astype(np.float32)
        self.phi = self.phi - self.turn_angle * turn_right.astype(np.float32)

        ties = (~turn_left) & (~turn_right) & (forward < left) & (forward < right)
        if np.any(ties):
            jitter = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=int(np.sum(ties)))
            self.phi[ties] += jitter * self.turn_angle * 0.5

        self.phi %= 2.0 * np.pi

        self._diffuse_and_decay()
        self.step_count += 1

    def run_steps(self, n_steps: int) -> None:
        for _ in range(n_steps):
            if self.step_count >= self.cfg.steps:
                break
            self.step()


def make_animation(cfg: Config, sim: PhysarumSimulation) -> tuple[plt.Figure, FuncAnimation]:
    frames = int(np.ceil(cfg.steps / cfg.updates_per_frame))

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    initial_data = np.log1p(sim.trail)
    image = ax.imshow(initial_data, cmap=cfg.cmap, vmin=0.0, vmax=1.0, animated=True)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(_frame: int):
        sim.run_steps(cfg.updates_per_frame)
        frame_data = np.log1p(sim.trail)
        vmax = float(np.percentile(frame_data, 99.5))
        image.set_data(frame_data)
        image.set_clim(0.0, max(vmax, 1e-6))
        ax.set_title(
            f"Physarum Simulation | step={sim.step_count}/{cfg.steps} | "
            f"agents={cfg.agents} | grid={cfg.grid_size}"
        )
        return [image]

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=cfg.interval_ms,
        blit=True,
        repeat=False,
    )
    return fig, anim


def main() -> None:
    cfg = parse_args()
    sim = PhysarumSimulation(cfg)

    print(
        f"Running Physarum with agents={cfg.agents}, grid={cfg.grid_size}, steps={cfg.steps}, "
        f"torus_wrap=True"
    )

    fig, animation = make_animation(cfg, sim)

    if cfg.save:
        lower = cfg.save.lower()
        if lower.endswith(".gif") or lower.endswith(".mp4"):
            animation.save(cfg.save, dpi=120)
            print(f"Saved animation to: {cfg.save}")
        else:
            remaining = cfg.steps - sim.step_count
            if remaining > 0:
                sim.run_steps(remaining)
            final_data = np.log1p(sim.trail)
            vmax = float(np.percentile(final_data, 99.5))
            fig.axes[0].images[0].set_data(final_data)
            fig.axes[0].images[0].set_clim(0.0, max(vmax, 1e-6))
            fig.savefig(cfg.save, dpi=150)
            print(f"Saved final frame to: {cfg.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
