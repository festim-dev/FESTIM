from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    Constant,
    dirichletbc,
    Function,
    FunctionSpace,
    locate_dofs_topological,
    locate_dofs_geometrical,
    form,
    assemble_scalar,
)
from dolfinx.fem.petsc import (
    NonlinearProblem,
)
from dolfinx.nls.petsc import NewtonSolver
from ufl import (
    dot,
    FiniteElement,
    grad,
    TestFunction,
    exp,
    FacetNormal,
    dx,
    ds,
    Cell,
    Mesh,
    VectorElement,
    Measure,
)
from dolfinx.mesh import create_mesh, meshtags, locate_entities
import numpy as np
import tqdm.autonotebook
import time
from test_permeation_problem import test_permeation_problem


def fenics_test_permeation_problem():
    L = 3e-04
    indices = np.linspace(0, L, num=1001)
    gdim, shape, degree = 1, "interval", 1
    cell = Cell(shape, geometric_dimension=gdim)
    domain = Mesh(VectorElement("Lagrange", cell, degree))
    mesh_points = np.reshape(indices, (len(indices), 1))
    indexes = np.arange(mesh_points.shape[0])
    cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)
    my_mesh = create_mesh(MPI.COMM_WORLD, cells, mesh_points, domain)
    fdim = my_mesh.topology.dim - 1
    vdim = my_mesh.topology.dim
    n = FacetNormal(my_mesh)

    elements = FiniteElement("CG", my_mesh.ufl_cell(), 1)
    V = FunctionSpace(my_mesh, elements)
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

    mobile_xdmf = XDMFFile(MPI.COMM_WORLD, "mobile_concentration.xdmf", "w")
    mobile_xdmf.write_mesh(my_mesh)

    flux_values = []
    times = []
    t = 0
    progress = tqdm.autonotebook.tqdm(
        desc="Solving H transport problem", total=num_steps
    )
    for i in range(num_steps):
        progress.update(1)
        t += dt

        solver.solve(u)

        surface_flux = form(D * dot(grad(u), n) * ds(2))
        flux = assemble_scalar(surface_flux)
        flux_values.append(flux)
        times.append(t)

        mobile_xdmf.write_function(u, t)

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

    relative_error = np.abs((flux_values - analytical_flux) / analytical_flux)

    relative_error = relative_error[
        np.where(analytical_flux > 0.01 * np.max(analytical_flux))
    ]
    error = relative_error.mean()

    assert error < 0.01


def test_festim_vs_fenics_permeation_benchmark():
    repetitions = 10

    fenics_times = []
    for i in range(repetitions):
        start = time.time()
        fenics_test_permeation_problem()
        fenics_times.append(time.time() - start)
    fenics_time = np.mean(fenics_times)

    festim_times = []
    for i in range(repetitions):
        start = time.time()
        test_permeation_problem()
        festim_times.append(time.time() - start)
    festim_time = np.mean(festim_times)

    diff = (np.abs(fenics_time - festim_time) / ((fenics_time + festim_time) / 2)) * 100
    if diff > 20:
        raise ValueError(f"festim is {diff:.1f}% slower than fenics")


if __name__ == "__main__":
    test_festim_vs_fenics_permeation_benchmark()
