import os

from petsc4py import PETSc

import numpy as np
from dolfinx.fem import Constant
from ufl import exp

import festim as F
import tempfile


def relative_error_computed_to_analytical(
    D, permeability, computed_flux, L, times, P_up
):
    n_array = np.arange(1, 10000)[:, np.newaxis]
    summation = np.sum(
        (-1) ** n_array * np.exp(-((np.pi * n_array) ** 2) * float(D) / L**2 * times),
        axis=0,
    )
    analytical_flux = P_up**0.5 * permeability / L * (2 * summation + 1)

    # post processing
    analytical_flux = np.abs(analytical_flux)
    indices = np.where(analytical_flux > 0.1 * np.max(analytical_flux))
    analytical_flux = analytical_flux[indices]
    computed_flux = computed_flux[indices]

    # evaulate relative error compared to analytical solution
    relative_error = np.abs((computed_flux - analytical_flux) / analytical_flux)
    error = relative_error.mean()

    return error


def test_permeation_problem(mesh_size=1001):
    """Test running a problem with a mobile species permeating through a 1D
    domain, checks that the computed permeation flux matches the analytical
    solution"""

    # festim model
    L = 3e-04
    vertices = np.linspace(0, L, num=mesh_size)

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
            subdomain=left_surface,
            S_0=4.02e21,
            E_S=1.04,
            pressure=100,
            species="H",
        ),
    ]

    temp_dir = tempfile.TemporaryDirectory()
    outgassing_flux = F.SurfaceFlux(
        filename=f"{temp_dir.name}/outgassing_flux.txt",
        field=mobile_H,
        surface=right_surface,
    )
    my_model.exports = [
        F.XDMFExport(temp_dir.name + "/mobile_concentration.xdmf", field=mobile_H),
        outgassing_flux,
    ]

    my_model.settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
        max_iterations=30,
        final_time=50,
    )

    my_model.settings.stepsize = F.Stepsize(initial_value=1 / 20)

    my_model.initialise()

    my_model.solver.convergence_criterion = "incremental"
    ksp = my_model.solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "gamg"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    my_model.run()

    times = outgassing_flux.t
    flux_values = outgassing_flux.data

    # -------------------------- analytical solution -------------------------------------

    D = my_mat.get_diffusion_coefficient(my_mesh.mesh, my_model.temperature)

    S_0 = float(my_model.boundary_conditions[-1].S_0)
    E_S = float(my_model.boundary_conditions[-1].E_S)
    P_up = float(my_model.boundary_conditions[-1].pressure)
    S = S_0 * exp(-E_S / F.k_B / float(my_model.temperature))
    permeability = float(D) * S
    times = np.array(times)
    flux_values = np.array(np.abs(flux_values))

    error = relative_error_computed_to_analytical(
        D, permeability, flux_values, L, times, P_up
    )

    assert error < 0.01


def test_permeation_problem_multi_volume(tmp_path):
    """Same permeation problem as above but with 4 volume subdomains instead
    of 1"""

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
            subdomain=left_surface,
            S_0=4.02e21,
            E_S=1.04,
            pressure=100,
            species="H",
        ),
    ]
    outgassing_flux = F.SurfaceFlux(
        filename=os.path.join(tmp_path, "outgassing_flux.csv"),
        field=mobile_H,
        surface=right_surface,
    )
    my_model.exports = [
        F.VTXSpeciesExport(
            os.path.join(tmp_path, "mobile_concentration_h.bp"), field=mobile_H
        ),
        outgassing_flux,
    ]

    my_model.settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
        max_iterations=30,
        final_time=50,
    )

    my_model.settings.stepsize = F.Stepsize(initial_value=1 / 20)

    my_model.initialise()

    my_model.solver.convergence_criterion = "incremental"
    ksp = my_model.solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "gamg"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    my_model.run()

    times = outgassing_flux.t
    flux_values = outgassing_flux.data

    # ---------------------- analytical solution -----------------------------
    D = my_mat.get_diffusion_coefficient(my_mesh.mesh, temperature)

    S_0 = float(my_model.boundary_conditions[-1].S_0)
    E_S = float(my_model.boundary_conditions[-1].E_S)
    P_up = float(my_model.boundary_conditions[-1].pressure)
    S = S_0 * exp(-E_S / F.k_B / float(temperature))
    permeability = float(D) * S
    times = np.array(times)
    flux_values = np.array(np.abs(flux_values))

    error = relative_error_computed_to_analytical(
        D, permeability, flux_values, L, times, P_up
    )

    assert error < 0.01
