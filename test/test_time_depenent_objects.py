from petsc4py import PETSc
from dolfinx.fem import Constant
from ufl import exp
import numpy as np
import festim as F


def test_permeation_problem(mesh_size=1001):
    """Test running a problem with a mobile species permeating through a 1D 0.3mm domain
    asserting that the resulting concentration field is less than 1% different from a
    respecitive analytical solution"""

    # festim model
    my_model = F.HydrogenTransportProblem()
    L = 3e-04
    my_model.mesh = F.Mesh1D(np.linspace(0, L, num=mesh_size))
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
        F.DirichletBC(
            subdomain=left_surface, value=lambda t: 1e17 + 1e17 * t, species="H"
        ),
    ]
    my_model.exports = [F.XDMFExport("mobile_concentration.xdmf", field=mobile_H)]

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

    times, flux_values = my_model.run()
