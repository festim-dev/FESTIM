"""Transient AMR on a curved trapping front (2D).

Hydrogen enters through a short patch on the left edge of a square of tungsten and is
captured by deep traps, which at 300 K fill essentially irreversibly. The saturated
region therefore grows as a *curved* front spreading from the patch -- the thing a 1D
run cannot show, and the case where AMR starts to pay for itself, because refining a
one-cell-wide arc costs O(N) cells while a uniform mesh of the same resolution costs
O(N^2).

Uses the ``BaseMeshRefiner`` backend: the base mesh is kept for the whole run and
re-refined from scratch each adapt against a per-base-cell level field, which is how
regions the front has passed get their resolution back (DOLFINx cannot coarsen).

Run::

    python amr_prototype/demo_2d_front.py                 # AMR run + figure
    python amr_prototype/demo_2d_front.py --compare       # + uniform runs (slow)
    python amr_prototype/demo_2d_front.py --gif           # animation as well
"""

import argparse
import pathlib
import time

from mpi4py import MPI

import numpy as np
from adapters import BaseMeshRefiner, n_cells
from dolfinx import mesh as dmesh
from dolfinx import plot
from indicators import problem_indicator, threshold_mark
from restart import rebuild

import festim as F

# --------------------------------------------------------------------------------
# Physics -- same trapping model as the 1D demo
# --------------------------------------------------------------------------------
L = 4e-5  # square side (m)
PATCH = L / 4  # height of the inlet patch on the left edge (m)
TEMPERATURE = 300.0

D_0, E_D = 1.9e-7, 0.2
C_SURFACE = 1e21
N_TRAP = 1e25
K_0, E_K = 1e-22, 0.0  # front width sqrt(D / (K_0 * N_TRAP)) ~ 0.3 um
P_0, E_P = 1e13, 1.5

FINAL_TIME = 1e5
INITIAL_DT = 1.0

OUT = pathlib.Path(__file__).parent / "out"


def square(n: int) -> F.Mesh:
    """Uniform triangulation of the square, as a FESTIM mesh."""
    return F.Mesh(
        mesh=dmesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([L, L])],
            [n, n],
            dmesh.CellType.triangle,
        )
    )


def build_problem(mesh: F.Mesh) -> F.HydrogenTransportProblemDiscontinuous:
    """Fresh problem on ``mesh``. Subdomains are located geometrically, so FESTIM
    re-derives the meshtags and nothing has to be transferred across an adapt."""
    model = F.HydrogenTransportProblemDiscontinuous()
    model.mesh = mesh

    tungsten = F.Material(name="tungsten", D_0=D_0, E_D=E_D)
    bulk = F.VolumeSubdomain(
        id=1, material=tungsten, locator=lambda x: np.full(x.shape[1], True)
    )
    inlet = F.SurfaceSubdomain(
        id=1,
        locator=lambda x: np.logical_and(np.isclose(x[0], 0.0), x[1] < PATCH + 1e-14),
    )
    model.subdomains = [bulk, inlet]

    mobile = F.Species("mobile", subdomains=[bulk])
    trapped = F.Species("trapped", mobile=False, subdomains=[bulk])
    empty_traps = F.ImplicitSpecies(n=N_TRAP, others=[trapped], name="empty")
    model.species = [mobile, trapped]

    model.reactions = [
        F.ArrheniusReaction(
            reactant=[mobile, empty_traps],
            product=trapped,
            k_0=K_0,
            E_k=E_K,
            p_0=P_0,
            E_p=E_P,
            volume=bulk,
        )
    ]

    model.temperature = TEMPERATURE
    # every other boundary is left at the natural no-flux condition, so the only way
    # in or out is the patch
    model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=inlet, value=C_SURFACE, species=mobile)
    ]
    # atol is deliberately far below the 1e12 the 1D demo uses. A SNES absolute
    # tolerance is compared against a residual that carries a factor of the cell
    # measure, and a cell here has area ~5e-13 where a 1D cell had length 1e-6. At
    # atol=1e12 the entire trapped-species residual sat below the tolerance, so the
    # solver "converged" instantly to a state that does not satisfy the equations --
    # the trapped concentration froze at 0.1 n_trap and no front ever formed, with no
    # error reported. AMR sharpens the trap: every refinement shrinks the cell measure
    # further, so a tolerance that was adequate on the base mesh silently stops
    # constraining the solution as the mesh adapts.
    model.settings = F.Settings(
        atol=1e2,
        rtol=1e-10,
        transient=True,
        final_time=FINAL_TIME,
        stepsize=F.Stepsize(
            INITIAL_DT,
            growth_factor=1.2,
            cutback_factor=0.8,
            target_nb_iterations=4,
            max_stepsize=FINAL_TIME / 100,
        ),
    )
    model.show_progress_bar = False
    return model


# --------------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------------
def snapshot(problem) -> dict:
    """Everything matplotlib needs to draw the mesh and the trapped field.

    ``plot.vtk_mesh(V)`` is used rather than the mesh geometry because it returns the
    connectivity in *dof* order, which is what lines up with ``u.x.array``.
    """
    subdomain = problem.volume_subdomains[0]
    trapped = next(s for s in problem.species if s.name == "trapped")
    u = trapped.subdomain_to_post_processing_solution[subdomain]

    topology, _, geometry = plot.vtk_mesh(u.function_space)
    triangles = topology.reshape(-1, 4)[:, 1:]
    return {
        "t": float(problem.t.value),
        "x": geometry[:, 0].copy(),
        "y": geometry[:, 1].copy(),
        "triangles": triangles.copy(),
        "c": u.x.array.copy() / N_TRAP,
        "cells": n_cells(problem),
    }


def peak_trapped(problem) -> float:
    """Largest trapped concentration anywhere, as a fraction of ``n_trap``.

    The discriminating accuracy metric. Filled area is an integral and stays accurate
    even on a mesh that cannot resolve the front; what an under-resolved P1 mesh does is
    oscillate, pushing the trapped concentration above saturation.
    """
    subdomain = problem.volume_subdomains[0]
    trapped = next(s for s in problem.species if s.name == "trapped")
    u = trapped.subdomain_to_post_processing_solution[subdomain]
    local = float(u.x.array.max()) if u.x.array.size else 0.0
    return problem.mesh.mesh.comm.allreduce(local, op=MPI.MAX) / N_TRAP


def saturated_area(problem) -> float:
    """Area where the traps are more than half full -- the size of the filled region."""
    import ufl
    from dolfinx import fem

    subdomain = problem.volume_subdomains[0]
    trapped = next(s for s in problem.species if s.name == "trapped")
    u = trapped.subdomain_to_post_processing_solution[subdomain]
    indicator = ufl.conditional(ufl.gt(u, 0.5 * N_TRAP), 1.0, 0.0)
    form = fem.form(indicator * ufl.dx)
    return problem.mesh.mesh.comm.allreduce(fem.assemble_scalar(form))


# --------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------
def run_uniform(n: int, label: str, snapshot_times) -> dict:
    problem = build_problem(square(n))
    problem.initialise()
    problem.post_processing()

    history = {"t": [], "cells": [], "area": [], "peak": [], "wall": []}
    snapshots = []
    start = time.perf_counter()
    pending = list(snapshot_times)

    while problem.t.value < FINAL_TIME:
        problem.iterate()
        history["t"].append(float(problem.t.value))
        history["cells"].append(n_cells(problem))
        history["area"].append(saturated_area(problem))
        history["peak"].append(peak_trapped(problem))
        history["wall"].append(time.perf_counter() - start)
        if pending and problem.t.value >= pending[0]:
            snapshots.append(snapshot(problem))
            pending.pop(0)

    history["label"] = label
    history["snapshots"] = snapshots
    print(
        f"{label}: {n_cells(problem)} cells, {len(history['t'])} steps, "
        f"{history['wall'][-1]:.1f} s wall"
    )
    return history


def run_amr(
    n_base: int = 30,
    max_level: int = 3,
    refine_above: float = 0.1,
    coarsen_below: float = 0.02,
    buffer: float = 2e-6,
    adapt_every: int = 5,
    snapshot_times=(),
    species_names=("trapped",),
) -> dict:
    base = square(n_base)
    refiner = BaseMeshRefiner(base, max_level=max_level, buffer=buffer)

    problem = build_problem(base)
    problem.initialise()
    problem.post_processing()

    history = {"t": [], "cells": [], "area": [], "peak": [], "wall": []}
    snapshots = []
    start = time.perf_counter()
    pending = list(snapshot_times)
    step = 0
    remeshes = 0

    while problem.t.value < FINAL_TIME:
        problem.iterate()
        history["t"].append(float(problem.t.value))
        history["cells"].append(n_cells(problem))
        history["area"].append(saturated_area(problem))
        history["peak"].append(peak_trapped(problem))
        history["wall"].append(time.perf_counter() - start)
        step += 1

        if pending and problem.t.value >= pending[0]:
            snapshots.append(snapshot(problem))
            pending.pop(0)

        if step % adapt_every:
            continue

        eta = problem_indicator(
            problem, species_names=list(species_names), kind="variation"
        )
        refine_cells, coarsen_cells, splits = threshold_mark(
            eta,
            refine_above=refine_above,
            coarsen_below=coarsen_below,
            max_splits=max_level,
        )
        new_mesh, ft, ct = refiner.adapt(problem, refine_cells, coarsen_cells, splits)
        problem = rebuild(
            build_problem,
            new_mesh,
            problem,
            t=float(problem.t.value),
            dt=float(problem.dt.value),
            facet_meshtags=ft,
            volume_meshtags=ct,
        )
        remeshes += 1

    history["label"] = "amr"
    history["snapshots"] = snapshots
    history["remeshes"] = remeshes
    print(
        f"amr: {history['cells'][-1]} cells at the end (max "
        f"{max(history['cells'])}), {len(history['t'])} steps, {remeshes} remeshes, "
        f"{history['wall'][-1]:.1f} s wall"
    )
    return history


# --------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------
def plot_snapshots(history, filename: str, title: str):
    """One column per snapshot: the mesh on top, the trapped field below."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    snapshots = history["snapshots"]
    fig, axes = plt.subplots(
        2, len(snapshots), figsize=(3.2 * len(snapshots), 6.6), squeeze=False
    )

    for column, snap in enumerate(snapshots):
        tri = Triangulation(snap["x"], snap["y"], snap["triangles"])

        top = axes[0][column]
        top.triplot(tri, color="k", lw=0.15)
        top.set_title(f"t = {snap['t']:.0f} s\n{snap['cells']} cells", fontsize=10)

        bottom = axes[1][column]
        mesh_plot = bottom.tripcolor(
            tri, snap["c"], shading="gouraud", vmin=0.0, vmax=1.0, cmap="magma"
        )

        for ax in (top, bottom):
            ax.set_aspect("equal")
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.colorbar(
        mesh_plot, ax=axes[1].tolist(), label="trapped / $n_{trap}$", shrink=0.85
    )
    fig.suptitle(title)
    path = OUT / filename
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_comparison(histories):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for h in histories:
        axes[0].plot(h["t"], np.array(h["area"]) / L**2, label=h["label"])
        # only the excess above saturation counts: early on nothing is trapped
        # anywhere, and being far below n_trap is the correct answer
        axes[1].plot(
            h["t"], np.maximum(np.array(h["peak"]) - 1.0, 0.0), label=h["label"]
        )
        axes[2].plot(h["t"], h["cells"], label=h["label"])
        axes[3].plot(h["t"], h["wall"], label=h["label"])

    axes[0].set(
        xlabel="t (s)", ylabel="saturated fraction of the square", title="Filled area"
    )
    axes[1].set(
        xlabel="t (s)",
        ylabel="max(c) / $n_{trap}$ - 1",
        title="Overshoot above saturation",
    )
    axes[2].set(xlabel="t (s)", ylabel="cells", yscale="log", title="Mesh size")
    axes[3].set(xlabel="t (s)", ylabel="wall clock (s)", title="Cost")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "front_2d_comparison.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def compare(histories):
    reference = next(
        (h for h in histories if h["label"].startswith("uniform fine")), None
    )
    print(f"\n{'run':>26} {'cells':>16} {'wall':>8} {'area err':>9} {'overshoot':>10}")
    for h in histories:
        cells = (
            f"{min(h['cells'])}-{max(h['cells'])}"
            if min(h["cells"]) != max(h["cells"])
            else str(h["cells"][-1])
        )
        area_err = (
            abs(h["area"][-1] - reference["area"][-1]) / reference["area"][-1]
            if reference
            else float("nan")
        )
        print(
            f"{h['label']:>26} {cells:>16} {h['wall'][-1]:>7.1f}s "
            f"{area_err:>8.2%} {max(h['peak']) - 1.0:>9.2%}"
        )


def save_histories(histories):
    """Persist the traces so the figures can be redrawn without re-running."""
    import json

    slim = [{k: v for k, v in h.items() if k != "snapshots"} for h in histories]
    path = OUT / "front_2d_history.json"
    with open(path, "w") as fh:
        json.dump(slim, fh)
    print(f"wrote {path}")


def write_gif(history, filename="front_2d.gif"):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.tri import Triangulation

    snapshots = history["snapshots"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 4.4))

    def draw(index):
        snap = snapshots[index]
        tri = Triangulation(snap["x"], snap["y"], snap["triangles"])
        for ax in (left, right):
            ax.clear()
            ax.set_aspect("equal")
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_xticks([])
            ax.set_yticks([])
        left.triplot(tri, color="k", lw=0.15)
        left.set_title(f"mesh -- {snap['cells']} cells", fontsize=10)
        right.tripcolor(
            tri, snap["c"], shading="gouraud", vmin=0.0, vmax=1.0, cmap="magma"
        )
        right.set_title(f"trapped / $n_{{trap}}$ -- t = {snap['t']:.0f} s", fontsize=10)

    animation = FuncAnimation(fig, draw, frames=len(snapshots), interval=400)
    path = OUT / filename
    animation.save(path, writer="pillow", fps=3)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also run uniform coarse and fine meshes (slow)",
    )
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    if args.final_time is not None:
        globals()["FINAL_TIME"] = args.final_time

    OUT.mkdir(exist_ok=True)
    n_frames = 40 if args.gif else args.frames
    times = np.linspace(FINAL_TIME / n_frames, FINAL_TIME, n_frames)

    amr = run_amr(snapshot_times=times)
    if args.gif:
        write_gif(amr)
        amr["snapshots"] = amr["snapshots"][:: max(1, n_frames // args.frames)]
    plot_snapshots(
        amr, "front_2d_amr.png", "Adaptive mesh following the trapping front"
    )

    if not args.compare:
        return

    histories = [amr]
    # 30 is the AMR run's base mesh; 240 is the uniform mesh with the same cell size
    # as the AMR run reaches at max_level=3, which is the honest cost comparison
    for n, label in ((30, "uniform base (h=1.3e-6)"), (240, "uniform fine (h=1.7e-7)")):
        histories.append(run_uniform(n, label, times[-1:]))
    plot_comparison(histories)
    compare(histories)
    save_histories(histories)


if __name__ == "__main__":
    main()
