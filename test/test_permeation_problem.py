from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    Constant,
    dirichletbc,
    locate_dofs_topological,
    form,
    assemble_scalar,
)
from ufl import dot, grad, exp, FacetNormal
import numpy as np
import tqdm.autonotebook


import festim as F


def test_permeation_problem():
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

    my_model.temperature = 500

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=right_surface, value=0, species="H"),
        F.SievertsBC(
            subdomain=left_surface, S_0=4.02e21, E_S=1.04, pressure=100, species="H"
        ),
    ]

    my_model.initialise()

    D = my_mat.get_diffusion_coefficient(my_mesh.mesh, my_model.temperature)

    V = my_model.function_space
    u = mobile_H.solution

    # TODO this should be a property of Mesh
    n = FacetNormal(my_mesh.mesh)

    S_0 = 4.02e21
    E_S = 1.04
    P_up = 100

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

    # analytical solution
    S = S_0 * exp(-E_S / F.k_B / float(my_model.temperature))
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


def test_permeation_problem_multi_volume():
    L = 3e-04
    vertices = np.linspace(0, L, num=1001)

    my_mesh = F.Mesh1D(vertices)

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = my_mesh

    my_mat = F.Material(D_0=1.9e-7, E_D=0.2, name="my_mat")
    my_subdomain_1 = F.VolumeSubdomain1D(id=1, borders=[0, L / 4], material=my_mat)
    my_subdomain_2 = F.VolumeSubdomain1D(id=2, borders=[L / 4, L / 2], material=my_mat)
    my_subdomain_3 = F.VolumeSubdomain1D(
        id=3, borders=[L / 2, 3 * L / 4], material=my_mat
    )
    my_subdomain_4 = F.VolumeSubdomain1D(id=4, borders=[3 * L / 4, L], material=my_mat)
    left_surface = F.SurfaceSubdomain1D(id=1, x=0)
    right_surface = F.SurfaceSubdomain1D(id=2, x=L)
    my_model.subdomains = [
        my_subdomain_1,
        my_subdomain_2,
        my_subdomain_3,
        my_subdomain_4,
        left_surface,
        right_surface,
    ]

    mobile_H = F.Species("H")
    my_model.species = [mobile_H]

    temperature = Constant(my_mesh.mesh, 500.0)
    my_model.temperature = temperature

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=right_surface, value=0, species="H"),
        F.SievertsBC(
            subdomain=left_surface, S_0=4.02e21, E_S=1.04, pressure=100, species="H"
        ),
    ]

    my_model.initialise()

    D = my_mat.get_diffusion_coefficient(my_mesh.mesh, temperature)

    V = my_model.function_space
    u = mobile_H.solution

    # TODO this should be a property of Mesh
    n = FacetNormal(my_mesh.mesh)

    def siverts_law(T, S_0, E_S, pressure):
        S = S_0 * exp(-E_S / F.k_B / T)
        return S * pressure**0.5

    S_0 = 4.02e21
    E_S = 1.04
    P_up = 100

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

    # analytical solution
    S = S_0 * exp(-E_S / F.k_B / float(temperature))
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
