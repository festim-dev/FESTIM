import numpy as np
import festim as F
from petsc4py import PETSc
from dolfinx.fem import Constant
from ufl import exp
import os


def test_multispecies_permeation_problem():
    """Test running a problem with 2 mobile species permeating through a 1D
    0.3mm domain, with different diffusion coefficients, asserting that the
    resulting concentration fields are less than 1% different from their
    respecitive analytical solutions"""

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

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=right_surface, value=0, species="spe_1"),
        F.DirichletBC(subdomain=right_surface, value=0, species="spe_2"),
        F.SievertsBC(
            subdomain=left_surface, S_0=4.02e21, E_S=1.04, pressure=100, species="spe_1"
        ),
        F.SievertsBC(
            subdomain=left_surface, S_0=4.02e21, E_S=1.04, pressure=100, species="spe_2"
        ),
    ]
    my_model.settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
        max_iterations=30,
        final_time=10,
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

    times, flux_values = my_model.run()

    # ---------------------- analytical solutions -----------------------------
    D_spe_1 = my_mat.get_diffusion_coefficient(my_mesh.mesh, temperature, spe_1)
    D_spe_2 = my_mat.get_diffusion_coefficient(my_mesh.mesh, temperature, spe_2)
    S_0 = float(my_model.boundary_conditions[-1].S_0)
    E_S = float(my_model.boundary_conditions[-1].E_S)
    P_up = float(my_model.boundary_conditions[-1].pressure)
    S = S_0 * exp(-E_S / F.k_B / float(temperature))
    permeability_spe_1 = float(D_spe_1) * S
    permeability_spe_2 = float(D_spe_2) * S
    times = np.array(times)

    # ##### compute analyical solution for species 1 ##### #
    n_array = np.arange(1, 10000)[:, np.newaxis]

    summation_spe_1 = np.sum(
        (-1) ** n_array
        * np.exp(-((np.pi * n_array) ** 2) * float(D_spe_1) / L**2 * times),
        axis=0,
    )
    analytical_flux_spe_1 = (
        P_up**0.5 * permeability_spe_1 / L * (2 * summation_spe_1 + 1)
    )

    # post processing
    analytical_flux_spe_1 = np.abs(analytical_flux_spe_1)
    flux_values_spe_1 = np.array(np.abs(flux_values[0]))
    indices_spe_1 = np.where(
        analytical_flux_spe_1 > 0.1 * np.max(analytical_flux_spe_1)
    )

    analytical_flux_spe_1 = analytical_flux_spe_1[indices_spe_1]
    flux_values_spe_1 = flux_values_spe_1[indices_spe_1]

    # evaluate relative error compared to analytical solution
    relative_error_spe_1 = np.abs(
        (flux_values_spe_1 - analytical_flux_spe_1) / analytical_flux_spe_1
    )
    error_spe_1 = relative_error_spe_1.mean()

    # ##### compute analyical solution for species 2 ##### #
    summation_spe_2 = np.sum(
        (-1) ** n_array
        * np.exp(-((np.pi * n_array) ** 2) * float(D_spe_2) / L**2 * times),
        axis=0,
    )
    analytical_flux_spe_2 = (
        P_up**0.5 * permeability_spe_2 / L * (2 * summation_spe_2 + 1)
    )

    # post processing
    analytical_flux_spe_2 = np.abs(analytical_flux_spe_2)
    flux_values_spe_2 = np.array(np.abs(flux_values[1]))
    indices_spe_2 = np.where(
        analytical_flux_spe_2 > 0.1 * np.max(analytical_flux_spe_2)
    )

    analytical_flux_spe_2 = analytical_flux_spe_2[indices_spe_2]
    flux_values_spe_2 = flux_values_spe_2[indices_spe_2]

    # evaluate relative error compared to analytical solution
    relative_error_spe_2 = np.abs(
        (flux_values_spe_2 - analytical_flux_spe_2) / analytical_flux_spe_2
    )
    error_spe_2 = relative_error_spe_2.mean()

    for err in [error_spe_1, error_spe_2]:
        assert err < 0.01
