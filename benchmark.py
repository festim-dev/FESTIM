import numpy as np
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
    MixedElement,
)
from dolfinx.mesh import create_mesh, meshtags, locate_entities
from dolfinx import log
import numpy as np
import tqdm.autonotebook
import festim as F
import pytest


def pure_fenics():
    # mesh nodes
    indices = np.linspace(0, 3e-4, num=1001)

    gdim, shape, degree = 1, "interval", 1
    cell = Cell(shape, geometric_dimension=gdim)
    domain = Mesh(VectorElement("Lagrange", cell, degree))
    mesh_points = np.reshape(indices, (len(indices), 1))
    indexes = np.arange(mesh_points.shape[0])
    cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)
    my_mesh = create_mesh(MPI.COMM_WORLD, cells, mesh_points, domain)
    fdim = my_mesh.topology.dim - 1
    vdim = my_mesh.topology.dim

    elements = FiniteElement("CG", my_mesh.ufl_cell(), 1)
    V = FunctionSpace(my_mesh, elements)
    u = Function(V)
    u_n = Function(V)
    v = TestFunction(V)

    borders = [0, 3e-04]

    num_cells = my_mesh.topology.index_map(vdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    markers = np.full(num_cells, 1, dtype=np.int32)
    markers[
        locate_entities(
            my_mesh,
            vdim,
            lambda x: np.logical_and(x[0] >= borders[0], x[0] <= borders[1]),
        )
    ] = 1
    mesh_tags_volumes = meshtags(my_mesh, vdim, cells, markers)

    tags_volumes = np.array([1], dtype=np.int32)

    dofs_L = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
    dofs_R = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], indices[-1]))

    dofs_facets = np.array([dofs_L[0], dofs_R[0]], dtype=np.int32)
    tags_facets = np.array([1, 2], dtype=np.int32)

    facet_dimension = my_mesh.topology.dim - 1
    volume_dimension = my_mesh.topology.dim

    mesh_tags_facets = meshtags(my_mesh, facet_dimension, dofs_facets, tags_facets)
    # mesh_tags_volumes = meshtags(my_mesh, volume_dimension, dofs_volumes, tags_volumes)
    ds = Measure("ds", domain=my_mesh, subdomain_data=mesh_tags_facets)
    dx = Measure("dx", domain=my_mesh, subdomain_data=mesh_tags_volumes)

    temperature = 500
    k_B = 8.6173303e-5
    n = FacetNormal(my_mesh)

    def siverts_law(T, S_0, E_S, pressure):
        S = S_0 * exp(-E_S / k_B / T)
        return S * pressure**0.5

    fdim = my_mesh.topology.dim - 1
    left_facets = mesh_tags_facets.find(1)
    left_dofs = locate_dofs_topological(V, fdim, left_facets)
    right_facets = mesh_tags_facets.find(2)
    right_dofs = locate_dofs_topological(V, fdim, right_facets)

    surface_conc = siverts_law(T=temperature, S_0=4.02e21, E_S=1.04, pressure=100)
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

    f = Constant(my_mesh, (PETSc.ScalarType(0)))
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
    # log.set_log_level(log.LogLevel.INFO)

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

        no_iters, converged = solver.solve(u)

        surface_flux = form(D * dot(grad(u), n) * ds(2))
        flux = assemble_scalar(surface_flux)
        flux_values.append(flux)
        times.append(t)
        np.savetxt("outgassing_flux.txt", np.array(flux_values))
        np.savetxt("times.txt", np.array(times))

        mobile_xdmf.write_function(u, t)

        u_n.x.array[:] = u.x.array[:]

    mobile_xdmf.close()


def festim_script():
    L = 3e-04
    vertices = np.linspace(0, L, num=1001)

    my_mesh = F.Mesh1D(vertices)

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = my_mesh

    my_mat = F.Material(D_0=1.9e-7, E_D=0.2, name="my_mat")
    my_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=my_mat)
    left_surface = F.SurfaceSubdomain1D(id=1, x=0)
    right_surface = F.SurfaceSubdomain1D(id=2, x=L)
    my_model.subdomains = [my_subdomain, left_surface, right_surface]

    mobile_H = F.Species("H")
    my_model.species = [mobile_H]

    temperature = Constant(my_mesh.mesh, 500.0)
    my_model.temperature = temperature

    my_model.initialise()

    D = my_mat.get_diffusion_coefficient(my_mesh.mesh, temperature)

    V = my_model.function_space
    u = mobile_H.solution

    # TODO this should be a property of Mesh
    n = FacetNormal(my_mesh.mesh)

    def siverts_law(T, S_0, E_S, pressure):
        S = S_0 * exp(-E_S / F.k_B / T)
        return S * pressure**0.5

    fdim = my_mesh.mesh.topology.dim - 1
    left_facets = my_model.facet_meshtags.find(1)
    left_dofs = locate_dofs_topological(V, fdim, left_facets)
    right_facets = my_model.facet_meshtags.find(2)
    right_dofs = locate_dofs_topological(V, fdim, right_facets)

    S_0 = 4.02e21
    E_S = 1.04
    P_up = 100
    surface_conc = siverts_law(T=temperature, S_0=S_0, E_S=E_S, pressure=P_up)
    bc_sieverts = dirichletbc(
        Constant(my_mesh.mesh, PETSc.ScalarType(surface_conc)), left_dofs, V
    )
    bc_outgas = dirichletbc(Constant(my_mesh.mesh, PETSc.ScalarType(0)), right_dofs, V)
    my_model.boundary_conditions = [bc_sieverts, bc_outgas]
    my_model.create_solver()

    my_model.solver.convergence_criterion = "incremental"
    my_model.solver.rtol = 1e-10
    my_model.solver.atol = 1e10

    my_model.solver.report = True
    ksp = my_model.solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "gamg"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    mobile_xdmf = XDMFFile(MPI.COMM_WORLD, "mobile_concentration.xdmf", "w")
    mobile_xdmf.write_mesh(my_model.mesh.mesh)

    final_time = 50

    flux_values = []
    times = []
    t = 0
    progress = tqdm.autonotebook.tqdm(
        desc="Solving H transport problem", total=final_time
    )
    while t < final_time:
        progress.update(float(my_model.dt))
        t += float(my_model.dt)

        my_model.solver.solve(u)

        mobile_xdmf.write_function(u, t)

        surface_flux = form(D * dot(grad(u), n) * my_model.ds(2))
        flux = assemble_scalar(surface_flux)
        flux_values.append(flux)
        times.append(t)

        mobile_H.prev_solution.x.array[:] = u.x.array[:]

    mobile_xdmf.close()


def test_fenicsx_benchmark(benchmark):
    benchmark(pure_fenics)


def test_festim_benchmark(benchmark):
    benchmark(festim_script)
