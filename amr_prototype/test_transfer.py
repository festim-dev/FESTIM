"""Sanity checks for the AMR prototype.

Deliberately *not* under ``test/``: nothing in ``src/festim`` is touched by this
investigation, so these are not library tests. Run them directly::

    pytest amr_prototype/test_transfer.py
"""

from mpi4py import MPI

import numpy as np
import pytest
from adapters import BaseMeshRefiner, Mesh1DAdapter, n_cells, uniform_mesh_1d
from dolfinx import fem
from dolfinx import mesh as dmesh
from indicators import threshold_mark, variation_indicator
from restart import rebuild, total_inventory

import festim as F

L = 1e-4


def build_problem(mesh):
    """Minimal transient problem: one mobile species, fixed value on the left."""
    model = F.HydrogenTransportProblemDiscontinuous()
    model.mesh = mesh

    material = F.Material(name="mat", D_0=1.9e-7, E_D=0.2)
    slab = F.VolumeSubdomain1D(id=1, borders=[0, L], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=L)
    model.subdomains = [slab, left, right]

    mobile = F.Species("mobile", subdomains=[slab])
    model.species = [mobile]
    model.temperature = 500.0
    model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left, value=1e21, species=mobile),
        F.FixedConcentrationBC(subdomain=right, value=0.0, species=mobile),
    ]
    model.settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
        transient=True,
        final_time=100.0,
        stepsize=F.Stepsize(1.0),
    )
    model.show_progress_bar = False
    return model


@pytest.fixture
def solved_problem():
    problem = build_problem(uniform_mesh_1d(0.0, L, 200))
    problem.initialise()
    for _ in range(20):
        problem.iterate()
    problem.post_processing()
    return problem


def test_refinement_transfer_is_exact(solved_problem):
    """Interpolating a P1 field onto a strictly finer mesh loses nothing.

    Every new vertex sits on a segment of the old mesh where the field is linear, so
    the transfer is exact -- which is why the drift observed in the trapping demo comes
    entirely from the coarsening half.
    """
    problem = solved_problem
    before = total_inventory(problem, "mobile")

    every_cell = np.arange(n_cells(problem), dtype=np.int32)
    adapter = Mesh1DAdapter(problem, hmin=L / 100000, hmax=L)
    new_mesh, _, _ = adapter.adapt(problem, every_cell, np.empty(0, np.int32))

    refined = rebuild(
        build_problem, new_mesh, problem, t=float(problem.t.value), dt=1.0
    )
    assert n_cells(refined) == 2 * n_cells(problem)
    assert total_inventory(refined, "mobile") == pytest.approx(before, rel=1e-12)


def test_coarsening_transfer_drifts_but_stays_bounded(solved_problem):
    """Coarsening is where inventory is lost, and the loss is small but systematic."""
    problem = solved_problem
    before = total_inventory(problem, "mobile")

    every_cell = np.arange(n_cells(problem), dtype=np.int32)
    adapter = Mesh1DAdapter(problem, hmin=L / 100000, hmax=L)
    new_mesh, _, _ = adapter.adapt(problem, np.empty(0, np.int32), every_cell)

    coarsened = rebuild(
        build_problem, new_mesh, problem, t=float(problem.t.value), dt=1.0
    )
    assert n_cells(coarsened) < n_cells(problem)
    assert total_inventory(coarsened, "mobile") == pytest.approx(before, rel=1e-2)


def test_round_trip_keeps_the_solution_usable(solved_problem):
    """Refine, coarsen, refine again, and the run must still step forward."""
    problem = solved_problem
    every_cell = lambda p: np.arange(n_cells(p), dtype=np.int32)  # noqa: E731
    empty = np.empty(0, dtype=np.int32)

    for refine in (True, False, True):
        adapter = Mesh1DAdapter(problem, hmin=L / 100000, hmax=L)
        marks = (every_cell(problem), empty) if refine else (empty, every_cell(problem))
        new_mesh, _, _ = adapter.adapt(problem, *marks)
        problem = rebuild(
            build_problem, new_mesh, problem, t=float(problem.t.value), dt=1.0
        )

    problem.iterate()
    problem.post_processing()
    inventory = total_inventory(problem, "mobile")
    assert np.isfinite(inventory) and inventory > 0


def test_protected_vertices_survive_coarsening():
    """Subdomain borders must stay in the mesh or ``check_borders`` rejects it."""
    # The border is put in the vertex list explicitly rather than trusting linspace to
    # land on it: np.linspace(0, 1e-4, 101)[50] is 4.9999999999999996e-05, not 5e-05,
    # and VolumeSubdomain1D's locator (x >= border) then leaves the straddling cell
    # untagged, which trips the "all cells must be tagged" assert in define_meshtags.
    vertices = np.unique(np.concatenate([np.linspace(0.0, L, 101), [L / 2]]))
    mesh = F.Mesh1D(vertices)

    material = F.Material(name="mat", D_0=1.9e-7, E_D=0.2)
    a = F.VolumeSubdomain1D(id=1, borders=[0, L / 2], material=material)
    b = F.VolumeSubdomain1D(id=2, borders=[L / 2, L], material=material)

    class _Stub:
        mesh = None
        volume_subdomains = [a, b]

    stub = _Stub()
    stub.mesh = mesh

    n = mesh.mesh.topology.index_map(1).size_local
    every_cell = np.arange(n, dtype=np.int32)
    adapter = Mesh1DAdapter(stub, hmin=L / 100000, hmax=L)
    new_mesh, _, _ = adapter.adapt(stub, np.empty(0, np.int32), every_cell)

    assert new_mesh.vertices.size < mesh.vertices.size, "nothing was coarsened"
    assert np.isclose(new_mesh.vertices, L / 2).any(), "the interface vertex was merged"
    # the real check: FESTIM accepts the coarsened mesh
    new_mesh.check_borders([a, b])


def test_variation_indicator_reads_the_gradient():
    """The indicator must equal the relative change of the field across a cell."""
    mesh = uniform_mesh_1d(0.0, 1.0, 10)
    V = fem.functionspace(mesh.mesh, ("Lagrange", 1))
    u = fem.Function(V)
    u.interpolate(lambda x: 2.0 * x[0])  # change of 0.2 per cell, scale 2.0

    eta = variation_indicator(u, scale=2.0)
    assert np.allclose(eta, 0.1)


def test_threshold_mark_splits_deeply_enough():
    """A cell four times over the threshold asks for two bisections, not one."""
    eta = np.array([0.001, 0.06, 0.2, 0.9])
    refine, coarsen, splits = threshold_mark(eta, refine_above=0.05, coarsen_below=0.01)
    assert refine.tolist() == [1, 2, 3]
    assert coarsen.tolist() == [0]
    assert splits.tolist() == [1, 2, 5 - 1]  # ceil(log2(0.9/0.05)) == 5, clipped to 4


def test_base_mesh_refiner_coarsens_back_to_base():
    """The nD backend must return a region to *exactly* base resolution."""
    base = F.Mesh(mesh=dmesh.create_unit_square(MPI.COMM_WORLD, 8, 8))
    refiner = BaseMeshRefiner(base, max_level=3)

    class _Stub:
        pass

    stub = _Stub()
    stub.mesh = base

    def band(mesh, lo, hi):
        n = mesh.topology.index_map(2).size_local
        mid = dmesh.compute_midpoints(mesh, 2, np.arange(n, dtype=np.int32))
        return np.flatnonzero((mid[:, 0] > lo) & (mid[:, 0] < hi)).astype(np.int32)

    empty = np.empty(0, dtype=np.int32)
    base_h = base.mesh.h(2, np.arange(refiner.n_base, dtype=np.int32)).max()

    # refine a band on the left, three levels deep
    current = base
    for _ in range(3):
        cells = band(current.mesh, 0.0, 0.3)
        current, _, _ = refiner.adapt(stub, cells, empty)
        stub.mesh = current
    assert n_cells_of(current) > 8 * 8 * 2

    # now move the band right; the left must come back to base resolution
    for _ in range(4):
        cells = band(current.mesh, 0.6, 0.8)
        coarsen = band(current.mesh, 0.0, 0.5)
        current, _, _ = refiner.adapt(stub, cells, coarsen)
        stub.mesh = current

    n = current.mesh.topology.index_map(2).size_local
    h = current.mesh.h(2, np.arange(n, dtype=np.int32))
    mid = dmesh.compute_midpoints(current.mesh, 2, np.arange(n, dtype=np.int32))
    left = mid[:, 0] < 0.3
    assert h[left].max() == pytest.approx(base_h)
    assert h[(mid[:, 0] > 0.6) & (mid[:, 0] < 0.8)].min() < base_h / 4


def n_cells_of(festim_mesh) -> int:
    imap = festim_mesh.mesh.topology.index_map(festim_mesh.mesh.topology.dim)
    return festim_mesh.mesh.comm.allreduce(imap.size_local)


def test_indicator_is_cheap(solved_problem):
    """The indicator must not cost more than the timestep it is deciding about.

    Assembling a DG0 form instead of reading the dofmap cost 60 ms per adapt, which was
    the single largest cost in the trapping demo before it was replaced.
    """
    import time

    from indicators import problem_indicator

    start = time.perf_counter()
    for _ in range(10):
        problem_indicator(solved_problem, kind="variation")
    assert (time.perf_counter() - start) / 10 < 5e-3
