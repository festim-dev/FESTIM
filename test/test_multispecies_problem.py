from petsc4py import PETSc

import numpy as np
from dolfinx.fem import Constant
from ufl import exp

import festim as F


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


def test_multispecies_permeation_problem():
    """Test running a problem with 2 mobile species permeating through a 1D
    domain, with different diffusion coefficients, checks that the computed
    permeation flux matches the analytical solution"""

    # festim model
    L = 3e-04
    my_mesh = F.Mesh1D(np.linspace(0, L, num=1001))
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = my_mesh

    my_mat = F.Material(
        D_0={"spe_1": 1.9e-7, "spe_2": 3.8e-7},
        E_D={"spe_1": 0.2, "spe_2": 0.2},
        name="my_mat",
    )
    my_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=my_mat)
    left_surface = F.SurfaceSubdomain1D(id=1, x=0)
    right_surface = F.SurfaceSubdomain1D(id=2, x=L)
    my_model.subdomains = [
        my_subdomain,
        left_surface,
        right_surface,
    ]

    spe_1 = F.Species("spe_1")
    spe_2 = F.Species("spe_2")
    my_model.species = [spe_1, spe_2]

    temperature = Constant(my_mesh.mesh, 500.0)
    my_model.temperature = temperature
    pressure = 100

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=right_surface, value=0, species="spe_1"),
        F.DirichletBC(subdomain=right_surface, value=0, species="spe_2"),
        F.SievertsBC(
            subdomain=left_surface,
            S_0=4.02e21,
            E_S=1.04,
            pressure=pressure,
            species="spe_1",
        ),
        F.SievertsBC(
            subdomain=left_surface,
            S_0=5.0e21,
            E_S=1.2,
            pressure=pressure,
            species="spe_2",
        ),
    ]
    my_model.settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
        max_iterations=30,
        final_time=10,
    )
    my_model.settings.stepsize = F.Stepsize(initial_value=1 / 20)

    outgassing_flux_spe_1 = F.SurfaceFlux(
        field=spe_1,
        surface=right_surface,
    )
    outgassing_flux_spe_2 = F.SurfaceFlux(
        field=spe_2,
        surface=right_surface,
    )
    total_species_1 = F.TotalVolume(
        field=spe_1,
        volume=my_subdomain,
    )
    total_species_2 = F.TotalVolume(
        field=spe_2,
        volume=my_subdomain,
    )
    my_model.exports = [
        outgassing_flux_spe_1,
        outgassing_flux_spe_2,
        total_species_1,
        total_species_2,
    ]
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

    times = outgassing_flux_spe_1.t
    flux_values_spe_1 = outgassing_flux_spe_1.data
    flux_values_spe_2 = outgassing_flux_spe_2.data

    # ---------------------- analytical solutions -----------------------------

    # common values
    times = np.array(times)
    P_up = float(my_model.boundary_conditions[-1].pressure)

    # ##### compute analyical solution for species 1 ##### #
    D_spe_1 = my_mat.get_diffusion_coefficient(my_mesh.mesh, temperature, spe_1)
    S_0_spe_1 = float(my_model.boundary_conditions[-2].S_0)
    E_S_spe_1 = float(my_model.boundary_conditions[-2].E_S)
    S_spe_1 = S_0_spe_1 * exp(-E_S_spe_1 / F.k_B / float(temperature))
    permeability_spe_1 = float(D_spe_1) * S_spe_1
    flux_values_spe_1 = np.array(np.abs(flux_values_spe_1))

    error_spe_1 = relative_error_computed_to_analytical(
        D_spe_1, permeability_spe_1, flux_values_spe_1, L, times, P_up
    )

    # ##### compute analyical solution for species 2 ##### #
    D_spe_2 = my_mat.get_diffusion_coefficient(my_mesh.mesh, temperature, spe_2)
    S_0_spe_2 = float(my_model.boundary_conditions[-1].S_0)
    E_S_spe_2 = float(my_model.boundary_conditions[-1].E_S)
    S_spe_2 = S_0_spe_2 * exp(-E_S_spe_2 / F.k_B / float(temperature))
    permeability_spe_2 = float(D_spe_2) * S_spe_2
    flux_values_spe_2 = np.array(np.abs(flux_values_spe_2))

    error_spe_2 = relative_error_computed_to_analytical(
        D_spe_2, permeability_spe_2, flux_values_spe_2, L, times, P_up
    )

    for err in [error_spe_1, error_spe_2]:
        assert err < 0.01
