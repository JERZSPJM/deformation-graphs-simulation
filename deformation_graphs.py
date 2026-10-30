"""A small spring-network simulation of a sheet pulled in tension.

Run from this folder with ``python deformation_graphs.py``. The script saves a
four-panel figure and a CSV time series beside itself.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm


def make_network(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a rectangular grid with horizontal, vertical, and diagonal links."""
    x, y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    points = np.column_stack((x.ravel(), y.ravel()))
    edges: list[tuple[int, int]] = []

    def node(row: int, col: int) -> int:
        return row * nx + col

    for row in range(ny):
        for col in range(nx):
            here = node(row, col)
            if col + 1 < nx:
                edges.append((here, node(row, col + 1)))
            if row + 1 < ny:
                edges.append((here, node(row + 1, col)))
            if col + 1 < nx and row + 1 < ny:
                edges.append((here, node(row + 1, col + 1)))
            if col + 1 < nx and row > 0:
                edges.append((here, node(row - 1, col + 1)))

    edge_array = np.asarray(edges, dtype=int)
    rest_lengths = np.linalg.norm(
        points[edge_array[:, 1]] - points[edge_array[:, 0]], axis=1
    )
    return points, edge_array, rest_lengths


def grip_motion(time: float, max_extension: float, ramp_time: float) -> tuple[float, float]:
    """Cubic smooth-step loading followed by a hold; return extension and speed."""
    if time <= 0:
        return 0.0, 0.0
    if time >= ramp_time:
        return max_extension, 0.0
    phase = time / ramp_time
    extension = max_extension * (3 * phase**2 - 2 * phase**3)
    speed = max_extension * (6 * phase - 6 * phase**2) / ramp_time
    return extension, speed


def spring_forces(
    points: np.ndarray,
    edges: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nodal forces and current link lengths for Hookean springs."""
    starts, ends = edges[:, 0], edges[:, 1]
    delta = points[ends] - points[starts]
    lengths = np.linalg.norm(delta, axis=1)
    extensions = lengths - rest_lengths
    link_forces = (
        stiffness * extensions / np.maximum(lengths, np.finfo(float).eps)
    )[:, None] * delta

    forces = np.zeros_like(points)
    np.add.at(forces, starts, link_forces)
    np.add.at(forces, ends, -link_forces)
    return forces, lengths


def simulate(args: argparse.Namespace) -> dict[str, np.ndarray]:
    points0, edges, rest_lengths = make_network(args.nx, args.ny)
    points = points0.copy()
    velocity = np.zeros_like(points)

    grid_cols = np.tile(np.arange(args.nx), args.ny)
    fixed = np.flatnonzero(grid_cols == 0)
    grip = np.flatnonzero(grid_cols == args.nx - 1)
    free = np.flatnonzero((grid_cols != 0) & (grid_cols != args.nx - 1))

    specimen_length = float(args.nx - 1)
    max_extension = args.strain * specimen_length
    total_time = args.ramp + args.hold
    steps = int(np.ceil(total_time / args.dt))

    # Keep every free node's mass identical so the model stays easy to inspect.
    point_mass = args.mass
    forces, lengths = spring_forces(points, edges, rest_lengths, args.stiffness)

    history: dict[str, list[float]] = {
        key: []
        for key in (
            "time_s",
            "grip_extension",
            "strain",
            "force",
            "power",
            "work",
            "spring_energy",
            "kinetic_energy",
            "dissipated_energy",
            "energy_balance_error",
        )
    }
    input_work = 0.0
    dissipated = 0.0
    previous_power = 0.0
    previous_dissipation_rate = 0.0

    def record(time: float, link_lengths: np.ndarray, nodal_forces: np.ndarray) -> tuple[float, float]:
        nonlocal input_work, dissipated, previous_power, previous_dissipation_rate
        extension, grip_speed = grip_motion(time, max_extension, args.ramp)
        reaction = -float(np.sum(nodal_forces[grip, 0]))
        power = reaction * grip_speed
        kinetic = 0.5 * point_mass * float(np.sum(velocity[free] ** 2))
        spring_energy = 0.5 * args.stiffness * float(
            np.sum((link_lengths - rest_lengths) ** 2)
        )
        dissipation_rate = args.damping * float(np.sum(velocity[free] ** 2))

        if history["time_s"]:
            elapsed = time - history["time_s"][-1]
            input_work += 0.5 * (previous_power + power) * elapsed
            dissipated += 0.5 * (previous_dissipation_rate + dissipation_rate) * elapsed

        values = {
            "time_s": time,
            "grip_extension": extension,
            "strain": extension / specimen_length,
            "force": reaction,
            "power": power,
            "work": input_work,
            "spring_energy": spring_energy,
            "kinetic_energy": kinetic,
            "dissipated_energy": dissipated,
            "energy_balance_error": input_work - spring_energy - kinetic - dissipated,
        }
        for key, value in values.items():
            history[key].append(float(value))
        previous_power = power
        previous_dissipation_rate = dissipation_rate
        return reaction, power

    record(0.0, lengths, forces)

    for step in range(steps):
        time = step * args.dt
        next_time = (step + 1) * args.dt

        # Semi-implicit Euler: spring forces update velocity, then position.
        acceleration = (forces[free] - args.damping * velocity[free]) / point_mass
        velocity[free] += acceleration * args.dt
        points[free] += velocity[free] * args.dt

        extension, grip_speed = grip_motion(next_time, max_extension, args.ramp)
        points[fixed, 0] = points0[fixed, 0]
        points[grip, 0] = points0[grip, 0] + extension
        velocity[fixed] = 0.0
        velocity[grip, 0] = grip_speed
        velocity[grip, 1] = 0.0

        forces, lengths = spring_forces(points, edges, rest_lengths, args.stiffness)
        record(next_time, lengths, forces)

    return {key: np.asarray(values) for key, values in history.items()} | {
        "initial_points": points0,
        "final_points": points,
        "edges": edges,
        "rest_lengths": rest_lengths,
        "nx": np.asarray([args.nx]),
        "ny": np.asarray([args.ny]),
    }


def save_csv(data: dict[str, np.ndarray], path: Path) -> None:
    columns = [
        "time_s",
        "grip_extension",
        "strain",
        "force",
        "power",
        "work",
        "spring_energy",
        "kinetic_energy",
        "dissipated_energy",
        "energy_balance_error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        writer.writerows(zip(*(data[column] for column in columns)))


def make_figure(data: dict[str, np.ndarray], path: Path, stiffness: float) -> None:
    points0 = data["initial_points"]
    points = data["final_points"]
    edges = data["edges"]
    rest_lengths = data["rest_lengths"]
    nx, ny = int(data["nx"][0]), int(data["ny"][0])
    starts, ends = edges[:, 0], edges[:, 1]
    current_lengths = np.linalg.norm(points[ends] - points[starts], axis=1)
    edge_strain = 100 * (current_lengths - rest_lengths) / rest_lengths

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    ax_graph, ax_energy, ax_power, ax_response = axes.flat

    # Initial mesh is a quiet reference; color on the final graph shows link strain.
    original_segments = np.stack((points0[starts], points0[ends]), axis=1)
    deformed_segments = np.stack((points[starts], points[ends]), axis=1)
    ax_graph.add_collection(
        LineCollection(original_segments, colors="#b9c1ca", linewidths=0.55, linestyles="dashed")
    )
    limit = max(float(np.max(np.abs(edge_strain))), 0.1)
    strain_norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    links = LineCollection(
        deformed_segments,
        cmap="coolwarm",
        norm=strain_norm,
        linewidths=1.1,
    )
    links.set_array(edge_strain)
    ax_graph.add_collection(links)
    ax_graph.scatter(points[:, 0], points[:, 1], s=8, color="#263746", zorder=3)
    ax_graph.set_xlim(-0.7, nx + 1.1)
    ax_graph.set_ylim(-0.8, ny - 0.2)
    ax_graph.set_aspect("equal", adjustable="box")
    ax_graph.set_xlabel("x (grid units)")
    ax_graph.set_ylabel("y (grid units)")
    ax_graph.set_title("Deformation graph · final link strain")
    colorbar = fig.colorbar(links, ax=ax_graph, fraction=0.046, pad=0.04)
    colorbar.set_label("Link strain (%)")
    ax_graph.text(0.02, 0.02, "Dashed: initial  |  Solid: deformed", transform=ax_graph.transAxes, fontsize=8)

    time = data["time_s"]
    ax_energy.plot(time, data["work"], label="Input work", color="#263746", linewidth=2)
    ax_energy.plot(time, data["spring_energy"], label="Spring energy", color="#d97732")
    ax_energy.plot(time, data["kinetic_energy"], label="Kinetic energy", color="#278b8e")
    ax_energy.plot(time, data["dissipated_energy"], label="Dissipated", color="#7866a6")
    ax_energy.plot(
        time,
        data["energy_balance_error"],
        label="Balance error",
        color="#888888",
        linewidth=0.9,
        linestyle="--",
    )
    ax_energy.set_title("Work and energy")
    ax_energy.set_xlabel("Time (s)")
    ax_energy.set_ylabel("Energy (model units)")
    ax_energy.grid(alpha=0.25)
    ax_energy.legend(frameon=False, fontsize=8, ncol=2)

    ax_power.plot(time, data["power"], color="#3274a1", linewidth=1.5)
    ax_power.fill_between(time, 0, data["power"], color="#3274a1", alpha=0.14)
    ax_power.axhline(0, color="#666666", linewidth=0.7)
    ax_power.set_title("Power at the moving grip")
    ax_power.set_xlabel("Time (s)")
    ax_power.set_ylabel("Power (model units)")
    ax_power.grid(alpha=0.25)

    stress = data["force"] / (ny - 1)  # 2D nominal stress, per unit thickness.
    ax_response.plot(100 * data["strain"], stress, color="#278b8e", linewidth=1.7)
    ax_response.scatter(100 * data["strain"][:: max(1, len(time) // 30)], stress[:: max(1, len(time) // 30)], s=11, color="#278b8e")
    ax_response.set_title("Tensile response")
    ax_response.set_xlabel("Grip strain (%)")
    ax_response.set_ylabel("Nominal stress (force / height)")
    ax_response.grid(alpha=0.25)
    ax_response.text(
        0.03,
        0.96,
        f"k = {stiffness:g} per link",
        transform=ax_response.transAxes,
        va="top",
        fontsize=8,
        color="#555555",
    )

    fig.suptitle("A spring network under uniaxial tension", fontsize=14)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a damped 2D spring network pulled by a moving grip."
    )
    parser.add_argument("--nx", type=int, default=15, help="number of grid columns (default: 15)")
    parser.add_argument("--ny", type=int, default=9, help="number of grid rows (default: 9)")
    parser.add_argument("--strain", type=float, default=0.10, help="target grip strain, e.g. 0.10 for 10%%")
    parser.add_argument("--ramp", type=float, default=0.5, help="loading time in seconds")
    parser.add_argument("--hold", type=float, default=0.5, help="hold time after loading in seconds")
    parser.add_argument("--dt", type=float, default=0.001, help="integration time step in seconds")
    parser.add_argument("--stiffness", type=float, default=40.0, help="spring stiffness per link")
    parser.add_argument("--damping", type=float, default=1.5, help="viscous damping coefficient")
    parser.add_argument("--mass", type=float, default=1.0, help="mass of each free node")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("deformation_simulation.png"),
        help="figure output path (CSV is saved beside it)",
    )
    parser.add_argument("--show", action="store_true", help="open the plot window after saving")
    args = parser.parse_args()
    if args.nx < 4 or args.ny < 3:
        parser.error("--nx must be at least 4 and --ny at least 3")
    if args.strain < 0 or args.strain > 0.5:
        parser.error("--strain must be between 0 and 0.5")
    if args.ramp <= 0 or args.hold < 0 or args.dt <= 0:
        parser.error("--ramp and --dt must be positive; --hold cannot be negative")
    if args.stiffness <= 0 or args.mass <= 0 or args.damping < 0:
        parser.error("--stiffness and --mass must be positive; --damping cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    data = simulate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_name(f"{args.output.stem}_timeseries.csv")
    make_figure(data, args.output, args.stiffness)
    save_csv(data, csv_path)
    final_error = data["energy_balance_error"][-1]
    print(f"Figure saved to: {args.output}")
    print(f"Time series saved to: {csv_path}")
    print(f"Final grip strain: {100 * data['strain'][-1]:.2f}%")
    print(f"Final input work: {data['work'][-1]:.4g} model energy units")
    print(f"Final energy-balance residual: {final_error:.4g}")
    if args.show:
        image = plt.imread(args.output)
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
