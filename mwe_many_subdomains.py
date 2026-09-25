"""MWE: HydrogenTransportProblemDiscontinuous.initialise() cost vs number of subdomains.

The unit square is cut into N vertical strips, each its own volume subdomain with its
own species, and the N - 1 interior strip boundaries form ONE codimension-1 network
subdomain that exchanges with every strip (the setting of a polycrystal with one
subdomain per grain; cf. test/system_tests/test_codim1_polycrystal.py, which this
generalises from 3 strips to N).

The mesh is the same for every N, so only the number of subdomains changes. Each
residual row of a strip depends on at most two unknowns (its own and the network's),
but create_formulation differentiates every row with respect to all N + 1 unknowns and,
because a manifold is present, expands every block before pruning the empty ones.

Everything is uniform in y, so the steady state is a 1D series chain with a closed form,
which is checked so that both code paths are seen to give the right answer:

    c=2 | strip 1 | net | strip 2 | ... | net | strip N | c=0
    J = 2 / (1/D + 2 (N - 1)/k)

Run it once on each branch and compare:

    python mwe_many_subdomains.py            # N = 4 8 16 32 64 128
    python mwe_many_subdomains.py 16 64      # chosen N
"""

import sys
import time
from contextlib import contextmanager

from mpi4py import MPI

import dolfinx
import numpy as np
import ufl

import festim as F
import festim.hydrogen_transport_problem as htp

D_BULK, D_NET, K = 1.5, 0.7, 5.0
CELLS = 256  # cells per side; every N must divide CELLS / 2 (strips >= 2 cells wide)
NETWORK_ID = 10_000


def build(n_strips):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, CELLS, CELLS)
    w, tol = 1.0 / n_strips, 1e-12
    strips = [
        F.VolumeSubdomain(
            id=i + 1,
            material=F.Material(D_0=D_BULK, E_D=0.0),
            locator=lambda x, a=i * w, b=(i + 1) * w: (
                (x[0] >= a - tol) & (x[0] <= b + tol)
            ),
        )
        for i in range(n_strips)
    ]

    def on_network(x):
        s = x[0] * n_strips
        return np.isclose(s, np.round(s)) & (x[0] > tol) & (x[0] < 1 - tol)

    network = F.VolumeSubdomain(
        id=NETWORK_ID,
        material=F.Material(D_0=D_NET, E_D=0.0),
        dim=1,
        locator=on_network,
    )
    left = F.SurfaceSubdomain(id=NETWORK_ID + 1, locator=lambda x: np.isclose(x[0], 0))
    right = F.SurfaceSubdomain(id=NETWORK_ID + 2, locator=lambda x: np.isclose(x[0], 1))

    species = [F.Species(f"c_{s.id}", subdomains=[s]) for s in strips]
    c_net = F.Species("c_net", subdomains=[network])
    sources = [
        F.ParticleSource(
            value=lambda c_g, c_b: K * (c_b - c_g),
            species=c_net,
            volume=network,
            species_dependent_value={"c_b": spe, "c_g": c_net},
        )
        for spe in species
    ]
    bcs = [
        F.ParticleFluxBC(
            subdomain=network,
            species=spe,
            value=lambda c_g, c_b: K * (c_g - c_b),
            species_dependent_value={"c_b": spe, "c_g": c_net},
        )
        for spe in species
    ]
    bcs += [
        F.FixedConcentrationBC(subdomain=left, value=2.0, species=species[0]),
        F.FixedConcentrationBC(subdomain=right, value=0.0, species=species[-1]),
    ]
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[*species, c_net],
        subdomains=[*strips, network, left, right],
        sources=sources,
        boundary_conditions=bcs,
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    return model, network, c_net


def network_error(n_strips, network, c_net):
    """Max deviation of the network concentration from the series-chain solution."""
    j = 2.0 / (1 / D_BULK + 2 * (n_strips - 1) / K)
    gb = c_net.subdomain_to_post_processing_solution[network]
    branch = np.rint(gb.function_space.tabulate_dof_coordinates()[:, 0] * n_strips)
    # branch b sits after b strips and 2b - 1 exchanges
    exact = 2.0 - j * branch / (n_strips * D_BULK) - (2 * branch - 1) * j / K
    return np.abs(gb.x.array - exact).max()


@contextmanager
def jacobian_symbolics():
    """Count the Jacobian blocks built and time the symbolic step that builds them:
    every ufl.derivative call plus prune_empty_blocks (expand, then drop zeros)."""
    stats = {"blocks": 0, "seconds": 0.0}
    real_derivative, real_prune = ufl.derivative, htp.prune_empty_blocks

    def derivative(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return real_derivative(*args, **kwargs)
        finally:
            stats["blocks"] += 1
            stats["seconds"] += time.perf_counter() - t0

    def prune(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return real_prune(*args, **kwargs)
        finally:
            stats["seconds"] += time.perf_counter() - t0

    ufl.derivative, htp.prune_empty_blocks = derivative, prune
    try:
        yield stats
    finally:
        ufl.derivative, htp.prune_empty_blocks = real_derivative, real_prune


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [4, 8, 16, 32, 64, 128]
    assert all((CELLS // 2) % n == 0 for n in ns), f"each N must divide {CELLS // 2}"
    print(f"FESTIM {F.__version__}, {CELLS}x{CELLS} unit square, warm FFCx cache")
    print(
        f"{'N':>5} {'blocks built':>13} {'Jacobian symbolics [s]':>23} "
        f"{'initialise [s]':>15} {'solve [s]':>10} {'max |c_net - exact|':>20}"
    )
    for n in ns:
        build(n)[0].initialise()  # compile once: both branches compile the same forms
        model, network, c_net = build(n)
        with jacobian_symbolics() as jac:
            t0 = time.perf_counter()
            model.initialise()
            t_init = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.run()
        t_solve = time.perf_counter() - t0
        print(
            f"{n:>5} {jac['blocks']:>13} {jac['seconds']:>23.2f} {t_init:>15.2f} "
            f"{t_solve:>10.2f} {network_error(n, network, c_net):>20.2e}",
            flush=True,
        )
