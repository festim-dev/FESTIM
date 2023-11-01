import festim as F
import numpy as np
from petsc4py import PETSc


def surface_flux_export():
    """Test that the field attribute can be a string and is found in the species list"""
    L = 3e-04
    my_mat = F.Material(D_0=1.9e-7, E_D=0.2, name="my_mat")
    my_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=my_mat)
    left_surface = F.SurfaceSubdomain1D(id=1, x=0)
    right_surface = F.SurfaceSubdomain1D(id=2, x=L)
    mobile_H = F.Species("H")

    my_model = F.HydrogenTransportProblem()

    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, L, num=1001))
    my_model.subdomains = [my_subdomain, left_surface, right_surface]
    my_model.species = [mobile_H]
    my_model.temperature = 500
    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=right_surface, value=0, species="H"),
        F.SievertsBC(
            subdomain=left_surface, S_0=4.02e21, E_S=1.04, pressure=100, species="H"
        ),
    ]

    my_model.settings = F.Settings(atol=1e10, rtol=1e-10, final_time=50)
    my_model.settings.stepsize = F.Stepsize(initial_value=1 / 20)

    my_export = F.SurfaceFlux(
        filename="my_surface_flux.csv",
        field=mobile_H,
        surface_subdomain=right_surface,
        volume_subdomain=my_subdomain,
    )
    my_model.exports = [my_export, F.XDMFExport("mobile_H.xdmf", field=mobile_H)]

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


if __name__ == "__main__":
    surface_flux_export()
