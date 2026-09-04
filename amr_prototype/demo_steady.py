"""Steady-state AMR: solve, estimate, refine, resolve.

The FESTIM analogue of the DOLFINx AMR tutorial
(https://jsdokken.com/dolfinx-tutorial/chapter2/amr.html). No coarsening is needed --
each iteration only ever adds resolution -- so this is the easy half of the problem and
it uses the classic relative (Dorfler-style) marking rather than the absolute threshold
the transient demo needs.

The model is a slab with a strongly localised volumetric source: a narrow implantation
peak near the left surface, which a uniform mesh has to resolve everywhere to get right.
The quantity of interest is the outgassing flux at the left surface.

Run::

    python amr_prototype/demo_steady.py
"""

import argparse
import pathlib

import numpy as np
import ufl
from adapters import Mesh1DAdapter, n_cells, uniform_mesh_1d
from indicators import dorfler_mark, problem_indicator
from restart import rebuild

import festim as F

L = 1e-4  # slab thickness (m)
TEMPERATURE = 500.0
D_0, E_D = 1.9e-7, 0.2

SOURCE_PEAK = 1e-6  # implantation depth (m)
SOURCE_WIDTH = 2e-7  # implantation width (m) -- the feature AMR has to find
SOURCE_FLUX = 1e20  # incident flux (m-2 s-1)

OUT = pathlib.Path(__file__).parent / "out"


def build_problem(mesh: F.Mesh) -> F.HydrogenTransportProblemDiscontinuous:
    """Fresh problem on ``mesh``, with a narrow Gaussian implantation source."""
    model = F.HydrogenTransportProblemDiscontinuous()
    model.mesh = mesh

    tungsten = F.Material(name="tungsten", D_0=D_0, E_D=E_D)
    slab = F.VolumeSubdomain1D(id=1, borders=[0, L], material=tungsten)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=L)
    model.subdomains = [slab, left, right]

    mobile = F.Species("mobile", subdomains=[slab])
    model.species = [mobile]

    x = ufl.SpatialCoordinate(mesh.mesh)
    gaussian = (
        SOURCE_FLUX
        / (SOURCE_WIDTH * np.sqrt(2 * np.pi))
        * ufl.exp(-((x[0] - SOURCE_PEAK) ** 2) / (2 * SOURCE_WIDTH**2))
    )
    model.sources = [F.ParticleSource(value=gaussian, volume=slab, species=mobile)]

    model.temperature = TEMPERATURE
    model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left, value=0.0, species=mobile),
        F.FixedConcentrationBC(subdomain=right, value=0.0, species=mobile),
    ]
    model.settings = F.Settings(atol=1e10, rtol=1e-12, transient=False)
    model.show_progress_bar = False
    return model


def outgassing_flux(problem) -> float:
    """Diffusive flux leaving the left surface, the quantity of interest."""
    from dolfinx import fem

    subdomain = problem.volume_subdomains[0]
    species = problem.species[0]
    u = species.subdomain_to_post_processing_solution[subdomain]

    submesh = subdomain.submesh
    n = ufl.FacetNormal(submesh)
    D = subdomain.material.get_diffusion_coefficient(
        submesh, problem.temperature_fenics, species
    )
    ds = ufl.Measure("ds", domain=submesh, subdomain_data=subdomain.ft)
    form = fem.form(-ufl.dot(D * ufl.grad(u), n) * ds(1))
    return submesh.comm.allreduce(fem.assemble_scalar(form))


def run(
    n_base: int = 40,
    max_iterations: int = 12,
    theta_refine: float = 0.3,
    hmin: float = L / 200000,
    tolerance: float = 1e-4,
):
    """Solve-estimate-refine until the quantity of interest stops moving."""
    problem = build_problem(uniform_mesh_1d(0.0, L, n_base))
    problem.initialise()
    problem.run()

    history = []
    previous = None
    for iteration in range(max_iterations):
        qoi = outgassing_flux(problem)
        cells = n_cells(problem)
        change = abs(qoi - previous) / abs(qoi) if previous is not None else np.inf
        history.append(
            {"iteration": iteration, "cells": cells, "qoi": qoi, "change": change}
        )
        print(
            f"iteration {iteration:>2}: {cells:>6} cells   flux = {qoi:.8e}"
            f"   change = {change:.2e}"
        )

        if change < tolerance:
            print(f"converged after {iteration} refinements")
            break
        previous = qoi

        eta = problem_indicator(problem, kind="gradient_jump")
        # steady AMR only ever adds resolution, so nothing is marked for coarsening
        refine_cells, _ = dorfler_mark(
            eta, problem.mesh.mesh.comm, theta_refine=theta_refine, theta_coarsen=0.0
        )
        if refine_cells.size == 0:
            print("nothing left to refine")
            break

        adapter = Mesh1DAdapter(problem, hmin=hmin, hmax=L, max_cells=200_000)
        new_mesh, ft, ct = adapter.adapt(problem, refine_cells, np.empty(0, np.int32))

        # the previous solution is carried over as the Newton initial guess; for a
        # linear problem it saves nothing, but it is what a nonlinear one would want
        problem = rebuild(
            build_problem,
            new_mesh,
            problem,
            facet_meshtags=ft,
            volume_meshtags=ct,
        )
        problem.run()

    return problem, history


def uniform_study(cell_counts) -> list[tuple[int, float]]:
    """Same QoI on uniform meshes, to check the AMR answer is the right one."""
    results = []
    for n in cell_counts:
        problem = build_problem(uniform_mesh_1d(0.0, L, n))
        problem.initialise()
        problem.run()
        results.append((n, outgassing_flux(problem)))
        print(f"uniform {n:>6} cells: flux = {results[-1][1]:.8e}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--skip-uniform", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    print("adaptive refinement")
    _, history = run(max_iterations=args.max_iterations)

    if args.skip_uniform:
        return

    print("\nuniform refinement, for comparison")
    reference = uniform_study([40, 160, 640, 2560, 10240])

    amr_qoi = history[-1]["qoi"]
    ref_qoi = reference[-1][1]
    print(
        f"\nAMR reached {amr_qoi:.8e} with {history[-1]['cells']} cells;"
        f" the {reference[-1][0]}-cell uniform mesh gives {ref_qoi:.8e}"
        f" (difference {abs(amr_qoi - ref_qoi) / abs(ref_qoi):.2%})"
    )


if __name__ == "__main__":
    main()
