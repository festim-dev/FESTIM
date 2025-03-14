import time

from mpi4py import MPI
from petsc4py import PETSc

import basix
import numpy as np
import tqdm.autonotebook
from dolfinx.fem import (
    Constant,
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_geometrical,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    NonlinearProblem,
)
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_mesh, locate_entities, meshtags
from dolfinx.nls.petsc import NewtonSolver
from test_permeation_problem import test_permeation_problem
from ufl import (
    FacetNormal,
    Measure,
    Mesh,
    TestFunction,
    dot,
    exp,
    grad,
)
import tempfile


def fenics_test_permeation_problem(mesh_size=1001):
    L = 3e-04
    indices = np.linspace(0, L, num=mesh_size)
    gdim, shape, degree = 1, "interval", 1
    domain = Mesh(basix.ufl.element("Lagrange", shape, degree, shape=(gdim,)))
    mesh_points = np.reshape(indices, (len(indices), 1))
    indexes = np.arange(mesh_points.shape[0])
    cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)
    my_mesh = create_mesh(MPI.COMM_WORLD, cells, mesh_points, domain)
    fdim = my_mesh.topology.dim - 1
    vdim = my_mesh.topology.dim
    n = FacetNormal(my_mesh)

    elements = basix.ufl.element(
        basix.ElementFamily.P,
        my_mesh.basix_cell(),
        1,
        basix.LagrangeVariant.equispaced,
    )
    V = functionspace(my_mesh, elements)
    u = Function(V)
    u_n = Function(V)
    v = TestFunction(V)

    num_cells = my_mesh.topology.index_map(vdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    markers = np.full(num_cells, 1, dtype=np.int32)
    markers[
        locate_entities(
            my_mesh,
            vdim,
            lambda x: np.logical_and(x[0] >= indices[0], x[0] <= indices[1]),
        )
    ] = 1
    mesh_tags_volumes = meshtags(my_mesh, vdim, cells, markers)

    dofs_L = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
    dofs_R = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], indices[-1]))

    dofs_facets = np.array([dofs_L[0], dofs_R[0]], dtype=np.int32)
    tags_facets = np.array([1, 2], dtype=np.int32)

    mesh_tags_facets = meshtags(my_mesh, fdim, dofs_facets, tags_facets)
    mesh_tags_volumes = meshtags(my_mesh, vdim, cells, markers)
    ds = Measure("ds", domain=my_mesh, subdomain_data=mesh_tags_facets)
    dx = Measure("dx", domain=my_mesh, subdomain_data=mesh_tags_volumes)

    temperature = 500
    k_B = 8.6173303e-5

    def siverts_law(T, S_0, E_S, pressure):
        S = S_0 * exp(-E_S / k_B / T)
        return S * pressure**0.5

    left_facets = mesh_tags_facets.find(1)
    my_mesh.topology.create_connectivity(fdim, my_mesh.topology.dim)
    left_dofs = locate_dofs_topological(V, fdim, left_facets)
    right_facets = mesh_tags_facets.find(2)
    right_dofs = locate_dofs_topological(V, fdim, right_facets)

    S_0 = 4.02e21
    E_S = 1.04
    P_up = 100
    surface_conc = siverts_law(T=temperature, S_0=S_0, E_S=E_S, pressure=P_up)
    bc_sieverts = dirichletbc(
        Constant(my_mesh, PETSc.ScalarType(surface_conc)), left_dofs, V
    )
    bc_outgas = dirichletbc(Constant(my_mesh, PETSc.ScalarType(0)), right_dofs, V)
    bcs = [bc_sieverts, bc_outgas]

    D_0 = 1.9e-7
    E_D = 0.2
    D = D_0 * exp(-E_D / k_B / temperature)

    dt = 1 / 20
    final_time = 50
    num_steps = int(final_time / dt)

    F = dot(D * grad(u), grad(v)) * dx(1)
    F += ((u - u_n) / dt) * v * dx(1)

    problem = NonlinearProblem(F, u, bcs=bcs)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e10
    solver.report = True
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "gamg"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    temp_dir = tempfile.TemporaryDirectory()
    mobile_xdmf = XDMFFile(
        MPI.COMM_WORLD, f"{temp_dir.name}/mobile_concentration.xdmf", "w"
    )
    mobile_xdmf.write_mesh(my_mesh)

    flux_values = []
    times = []
    t = 0
    progress = tqdm.autonotebook.tqdm(
        desc="Solving H transport problem", total=final_time, unit_scale=True
    )
    while t < final_time:
        progress.update(float(dt))
        t += float(dt)

        solver.solve(u)

        mobile_xdmf.write_function(u, t)

        surface_flux = form(D * dot(grad(u), n) * ds(2))
        flux = assemble_scalar(surface_flux)
        flux_values.append(flux)
        times.append(t)

        u_n.x.array[:] = u.x.array[:]

    mobile_xdmf.close()

    # analytical solution
    S = S_0 * exp(-E_S / k_B / float(temperature))
    permeability = float(D) * S
    times = np.array(times)

    n_array = np.arange(1, 10000)[:, np.newaxis]
    summation = np.sum(
        (-1) ** n_array * np.exp(-((np.pi * n_array) ** 2) * float(D) / L**2 * times),
        axis=0,
    )
    analytical_flux = P_up**0.5 * permeability / L * (2 * summation + 1)

    analytical_flux = np.abs(analytical_flux)
    flux_values = np.array(np.abs(flux_values))

    indices = np.where(analytical_flux > 0.01 * np.max(analytical_flux))
    analytical_flux = analytical_flux[indices]
    flux_values = flux_values[indices]

    relative_error = np.abs((flux_values - analytical_flux) / analytical_flux)

    error = relative_error.mean()

    assert error < 0.01


def test_festim_vs_fenics_permeation_benchmark():
    """Runs a problem with pure fenicsx and the same problem with FESTIM and
    raise ValueError if difference is too high"""

    start = time.time()
    fenics_test_permeation_problem(mesh_size=20001)
    fenics_time = time.time() - start

    start = time.time()
    test_permeation_problem(mesh_size=20001)
    festim_time = time.time() - start

    diff = (fenics_time - festim_time) / fenics_time
    threshold = -0.1
    if diff < threshold:
        raise ValueError(
            f"festim is {
                np.abs(diff):.1%
            } slower than fenics, current acceptable threshold of {
                np.abs(threshold):.1%
            }"
        )
    else:
        print(f"avg relative diff between festim and fenics {diff:.1%}")


if __name__ == "__main__":
    test_festim_vs_fenics_permeation_benchmark()
