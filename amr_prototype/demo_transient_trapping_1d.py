"""Transient AMR on an advancing trapping front (1D).

A tungsten slab is held at a fixed mobile concentration on its left surface. Deep traps
fill essentially irreversibly at 300 K, so the trapped concentration is a step: fully
saturated behind the front, empty ahead of it. The front advances diffusively, roughly
as sqrt(t).

That is the case AMR should win on. The mesh needs to resolve a region a few microns
wide that moves across a 100 micron slab, and a uniform mesh either wastes cells
everywhere or fails to resolve the front. Refining alone is not enough: without
coarsening behind the front, the mesh only grows and the run gets slower over time.

Run::

    python amr_prototype/demo_transient_trapping_1d.py --mode amr
    python amr_prototype/demo_transient_trapping_1d.py --mode fine     # reference
    python amr_prototype/demo_transient_trapping_1d.py --mode coarse
    python amr_prototype/demo_transient_trapping_1d.py --mode all   # all + plots
"""

import argparse
import json
import pathlib
import time

import numpy as np
from adapters import Mesh1DAdapter, n_cells, uniform_mesh_1d
from indicators import problem_indicator, threshold_mark
from restart import gather_profile, rebuild, restart_needed, total_inventory

import festim as F

# --------------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------------
L = 1e-4  # slab thickness (m)
TEMPERATURE = 300.0  # K -- cold enough that detrapping is negligible

D_0, E_D = 1.9e-7, 0.2  # tungsten self-diffusion of H
C_SURFACE = 1e21  # imposed mobile concentration at x=0 (m-3)
N_TRAP = 1e25  # trap density (m-3)
# Trapping rate. The reaction front has width sqrt(D / (K_0 * N_TRAP)) ~ 0.3 um here,
# which is 300x thinner than the slab but still resolvable -- with the physical rate of
# a real deep trap the front is a nanometre wide and no mesh resolves it at all.
K_0, E_K = 1e-22, 0.0  # trapping rate (m3/s)
P_0, E_P = 1e13, 1.5  # detrapping (s-1) -- frozen out at 300 K

FINAL_TIME = 1e5
INITIAL_DT = 1.0

OUT = pathlib.Path(__file__).parent / "out"

HISTORY_KEYS = (
    "t",
    "cells",
    "front",
    "inventory",
    "overshoot",
    "undershoot",
    "wall",
)


def build_problem(mesh: F.Mesh) -> F.HydrogenTransportProblemDiscontinuous:
    """Build a fresh problem on ``mesh``.

    Every subdomain and species object is created here rather than shared, because they
    carry FE state (``submesh``, ``u``, ``cell_map``, ``subdomain_to_*``) that must not
    leak from one generation to the next.
    """
    model = F.HydrogenTransportProblemDiscontinuous()
    model.mesh = mesh

    tungsten = F.Material(name="tungsten", D_0=D_0, E_D=E_D)
    slab = F.VolumeSubdomain1D(id=1, borders=[0, L], material=tungsten)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=L)
    model.subdomains = [slab, left, right]

    mobile = F.Species("mobile", subdomains=[slab])
    trapped = F.Species("trapped", mobile=False, subdomains=[slab])
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
            volume=slab,
        )
    ]

    model.temperature = TEMPERATURE
    model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left, value=C_SURFACE, species=mobile),
        F.FixedConcentrationBC(subdomain=right, value=0.0, species=mobile),
    ]
    model.settings = F.Settings(
        atol=1e12,
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
# Diagnostics
# --------------------------------------------------------------------------------
def front_position(x: np.ndarray, c: np.ndarray) -> float:
    """Where the trapped profile crosses half saturation, by linear interpolation."""
    below = np.flatnonzero(c < 0.5 * N_TRAP)
    if below.size == 0:
        return float(x[-1])
    i = below[0]
    if i == 0:
        return float(x[0])
    c0, c1 = c[i - 1], c[i]
    if c0 == c1:
        return float(x[i])
    return float(x[i - 1] + (0.5 * N_TRAP - c0) / (c1 - c0) * (x[i] - x[i - 1]))


def record(history, problem, wall):
    """Append one row of diagnostics.

    ``overshoot`` is the discriminating one. Inventory and front position are
    integrated quantities and stay close even on a mesh that cannot resolve the front;
    what an under-resolved P1 mesh actually does is oscillate, driving the trapped
    concentration above saturation and below zero. That is both unphysical and what
    eventually kills the Newton solve.
    """
    x, c = gather_profile(problem, "trapped")
    history["t"].append(float(problem.t.value))
    history["cells"].append(n_cells(problem))
    history["front"].append(front_position(x, c))
    history["inventory"].append(total_inventory(problem, "trapped"))
    # only the excess above saturation counts: early on nothing is trapped anywhere,
    # and being far below n_trap is the correct answer, not an overshoot
    history["overshoot"].append(max(0.0, float(c.max()) / N_TRAP - 1.0))
    history["undershoot"].append(float(c.min()) / N_TRAP)
    history["wall"].append(wall)


# --------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------
def run_uniform(n: int, label: str) -> dict:
    """Fixed-mesh run, for reference and for comparison."""
    problem = build_problem(uniform_mesh_1d(0.0, L, n))
    problem.initialise()

    history = {k: [] for k in HISTORY_KEYS}
    start = time.perf_counter()
    problem.post_processing()
    record(history, problem, 0.0)

    while problem.t.value < FINAL_TIME:
        problem.iterate()
        record(history, problem, time.perf_counter() - start)

    x, c = gather_profile(problem, "trapped")
    history["profile_x"] = x.tolist()
    history["profile_c"] = c.tolist()
    history["label"] = label
    history["remeshes"] = 0
    print(
        f"{label}: {n} cells, {len(history['t'])} steps, "
        f"{history['wall'][-1]:.1f} s wall"
    )
    return history


def run_amr(
    n_base: int = 100,
    hmin: float = L / 20000,
    hmax: float = L / 50,
    refine_above: float = 0.05,
    coarsen_below: float = 0.002,
    adapt_every: int = 5,
    max_cells: int = 4000,
    dt_factor: float = 1.0,
    buffer: float = 3e-6,
) -> dict:
    """Adaptive run: iterate, and rebuild on the adapted mesh every few steps."""
    problem = build_problem(uniform_mesh_1d(0.0, L, n_base))
    problem.initialise()

    history = {k: [] for k in HISTORY_KEYS}
    start = time.perf_counter()
    problem.post_processing()
    record(history, problem, 0.0)

    previous_marks = None
    step = 0
    remeshes = 0

    while problem.t.value < FINAL_TIME:
        problem.iterate()
        record(history, problem, time.perf_counter() - start)
        step += 1

        if step % adapt_every:
            continue

        eta = problem_indicator(problem, kind="variation")
        refine_cells, coarsen_cells, splits = threshold_mark(
            eta, refine_above=refine_above, coarsen_below=coarsen_below
        )
        if not restart_needed(previous_marks, refine_cells, coarsen_cells):
            continue

        adapter = Mesh1DAdapter(
            problem, hmin=hmin, hmax=hmax, max_cells=max_cells, buffer=buffer
        )
        new_mesh, ft, ct = adapter.adapt(problem, refine_cells, coarsen_cells, splits)

        t_now = float(problem.t.value)
        # Resuming at the full timestep turns out to be fine: the transferred state is
        # a good Newton guess. Halving it here (dt_factor=0.5) cost 80% more steps and
        # bought nothing, because the adaptive stepsize then spent the next several
        # steps climbing back. dt_factor is kept as the knob to turn if a stiffer
        # problem does struggle on the first step after a remesh.
        dt_now = float(problem.dt.value) * dt_factor
        problem = rebuild(
            build_problem,
            new_mesh,
            problem,
            t=t_now,
            dt=dt_now,
            facet_meshtags=ft,
            volume_meshtags=ct,
        )
        previous_marks = (refine_cells, coarsen_cells)
        remeshes += 1

    x, c = gather_profile(problem, "trapped")
    history["profile_x"] = x.tolist()
    history["profile_c"] = c.tolist()
    history["label"] = "amr"
    history["remeshes"] = remeshes
    print(
        f"amr: {history['cells'][-1]} cells at the end (max "
        f"{max(history['cells'])}), {len(history['t'])} steps, {remeshes} remeshes, "
        f"{history['wall'][-1]:.1f} s wall"
    )
    return history


# --------------------------------------------------------------------------------
def plot(histories):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for h in histories:
        label = h["label"]
        axes[0].plot(h["profile_x"], np.array(h["profile_c"]) / N_TRAP, label=label)
        axes[1].plot(h["t"], h["overshoot"], label=label)
        axes[2].plot(h["t"], h["cells"], label=label)
        axes[3].plot(h["t"], h["wall"], label=label)

    front = max(h["front"][-1] for h in histories)
    axes[0].set(
        xlabel="x (m)",
        ylabel="trapped / $n_{trap}$",
        title=f"Final profile (t = {histories[0]['t'][-1]:.0f} s)",
        xlim=(0, 1.6 * front),
    )
    axes[0].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[1].set(
        xlabel="t (s)",
        ylabel="max(c) / $n_{trap}$ - 1",
        title="Overshoot above saturation",
    )
    axes[2].set(xlabel="t (s)", ylabel="cells", title="Mesh size")
    axes[3].set(xlabel="t (s)", ylabel="wall clock (s)", title="Cost")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "trapping_front.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def compare(histories):
    """Report each run against the fine reference."""
    fine = next((h for h in histories if h["label"] == "fine"), None)
    if fine is None:
        return

    ref_x = np.array(fine["profile_x"])
    ref_c = np.array(fine["profile_c"])
    print(
        f"\n{'run':>8} {'cells':>12} {'wall':>8} {'inventory':>11} "
        f"{'front':>9} {'overshoot':>10} {'profile L2':>11}"
    )
    for h in histories:
        x = np.array(h["profile_x"])
        c = np.array(h["profile_c"])
        # compare on the reference mesh, where the reference is exact
        interp = np.interp(ref_x, x, c)
        l2 = np.sqrt(np.trapezoid((interp - ref_c) ** 2, ref_x))
        l2 /= np.sqrt(np.trapezoid(ref_c**2, ref_x))
        inv_err = (
            abs(h["inventory"][-1] - fine["inventory"][-1]) / fine["inventory"][-1]
        )
        front_err = abs(h["front"][-1] - fine["front"][-1]) / fine["front"][-1]
        cells = f"{min(h['cells'])}-{max(h['cells'])}"
        print(
            f"{h['label']:>8} {cells:>12} {h['wall'][-1]:>7.1f}s "
            f"{inv_err:>10.2%} {front_err:>8.2%} "
            f"{max(h['overshoot']):>9.2%} {l2:>10.2%}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", default="all", choices=["amr", "fine", "coarse", "all"]
    )
    parser.add_argument("--fine-cells", type=int, default=2000)
    parser.add_argument("--coarse-cells", type=int, default=100)
    parser.add_argument("--final-time", type=float, default=None)
    args = parser.parse_args()

    if args.final_time is not None:
        globals()["FINAL_TIME"] = args.final_time

    OUT.mkdir(exist_ok=True)
    histories = []
    if args.mode in ("coarse", "all"):
        histories.append(run_uniform(args.coarse_cells, "coarse"))
    if args.mode in ("amr", "all"):
        histories.append(run_amr())
    if args.mode in ("fine", "all"):
        histories.append(run_uniform(args.fine_cells, "fine"))

    with open(OUT / f"trapping_front_{args.mode}.json", "w") as fh:
        json.dump(histories, fh)

    if len(histories) > 1:
        plot(histories)
        compare(histories)


if __name__ == "__main__":
    main()
