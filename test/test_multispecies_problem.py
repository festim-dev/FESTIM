import numpy as np
import festim as F
from petsc4py import PETSc
from dolfinx.fem import Constant
from ufl import exp


def test_multispecies_permeation_problem():
    L = 3e-04
    my_mesh = F.Mesh1D(np.linspace(0, L, num=1001))
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = my_mesh

    my_mat = F.Material(
        D_0={"D": 1.9e-7, "T": 3.8e-7},
        E_D={"D": 0.2, "T": 0.2},
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

    mobile_D = F.Species("D")
    mobile_T = F.Species("T")
    my_model.species = [mobile_D, mobile_T]

    temperature = Constant(my_mesh.mesh, 500.0)
    my_model.temperature = temperature

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=right_surface, value=0, species="D"),
        F.DirichletBC(subdomain=right_surface, value=0, species="T"),
        F.SievertsBC(
            subdomain=left_surface, S_0=4.02e21, E_S=1.04, pressure=100, species="D"
        ),
        F.SievertsBC(
            subdomain=left_surface, S_0=4.02e21, E_S=1.04, pressure=100, species="T"
        ),
    ]
    my_model.exports = [
        F.XDMFExport("mobile_concentration_D.xdmf", field=mobile_D),
        F.XDMFExport("mobile_concentration_T.xdmf", field=mobile_T),
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

    # ---------------------- analytical solution -----------------------------
    D_1 = my_mat.get_diffusion_coefficient(my_mesh.mesh, temperature, mobile_D)
    D_2 = my_mat.get_diffusion_coefficient(my_mesh.mesh, temperature, mobile_T)
    S_0 = float(my_model.boundary_conditions[-1].S_0)
    E_S = float(my_model.boundary_conditions[-1].E_S)
    P_up = float(my_model.boundary_conditions[-1].pressure)
    S = S_0 * exp(-E_S / F.k_B / float(temperature))
    permeability_1 = float(D_1) * S
    permeability_2 = float(D_2) * S
    times = np.array(times)

    n_array_1 = np.arange(1, 10000)[:, np.newaxis]
    n_array_2 = np.arange(1, 10000)[:, np.newaxis]
    summation_1 = np.sum(
        (-1) ** n_array_1
        * np.exp(-((np.pi * n_array_1) ** 2) * float(D_1) / L**2 * times),
        axis=0,
    )
    summation_2 = np.sum(
        (-1) ** n_array_2
        * np.exp(-((np.pi * n_array_2) ** 2) * float(D_2) / L**2 * times),
        axis=0,
    )
    analytical_flux_1 = P_up**0.5 * permeability_1 / L * (2 * summation_1 + 1)
    analytical_flux_2 = P_up**0.5 * permeability_2 / L * (2 * summation_2 + 1)
    analytical_flux_1 = np.abs(analytical_flux_1)
    analytical_flux_2 = np.abs(analytical_flux_2)
    flux_values_1 = np.array(np.abs(flux_values[0]))
    flux_values_2 = np.array(np.abs(flux_values[1]))

    indices_1 = np.where(analytical_flux_1 > 0.1 * np.max(analytical_flux_1))
    indices_2 = np.where(analytical_flux_2 > 0.1 * np.max(analytical_flux_2))
    analytical_flux_1 = analytical_flux_1[indices_1]
    analytical_flux_2 = analytical_flux_2[indices_2]
    flux_values_1 = flux_values_1[indices_1]
    flux_values_2 = flux_values_2[indices_2]

    relative_error_1 = np.abs((flux_values_1 - analytical_flux_1) / analytical_flux_1)
    relative_error_2 = np.abs((flux_values_2 - analytical_flux_2) / analytical_flux_2)

    error_1 = relative_error_1.mean()
    error_2 = relative_error_2.mean()

    for err in [error_1, error_2]:
        assert err < 0.01
