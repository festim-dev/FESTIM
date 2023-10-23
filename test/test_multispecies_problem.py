import numpy as np
import festim as F


def test_multispecies_problem_initialisation():
    """Test that the multispecies problem is correctly initialised"""
    my_model = F.HydrogenTransportProblem()
    L = 1e-04
    my_model.mesh = F.Mesh1D(np.linspace(0, L, num=1001))
    my_mat = F.Material(
        D_0={"H": 1.9e-7, "D": 1.9e-07}, E_D={"H": 0.2, "D": 0.2}, name="my_mat"
    )
    my_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=my_mat)
    left_surface = F.SurfaceSubdomain1D(id=1, x=0)
    right_surface = F.SurfaceSubdomain1D(id=2, x=L)
    my_model.subdomains = [my_subdomain, left_surface, right_surface]
    mobile_H = F.Species("H")
    mobile_D = F.Species("D")
    my_model.species = [mobile_H, mobile_D]
    my_model.temperature = 600
    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left_surface, value=1e18, species="H"),
        F.DirichletBC(subdomain=right_surface, value=1e18, species="D"),
    ]
    my_model.settings = F.Settings(atol=1e10, rtol=1e-10, final_time=50)
    my_model.settings.stepsize = 0.1
    my_model.exports = [
        F.XDMFExport("results/multispecies/test_H.xdmf", field=mobile_H),
        F.XDMFExport("results/multispecies/test_D.xdmf", field=mobile_D),
        F.VTXExport("results/multispecies/test_H_vts.bp", field=mobile_H),
        F.VTXExport("results/multispecies/test_D_vtx.bp", field=mobile_D),
    ]

    my_model.initialise()
    my_model.run()


if __name__ == "__main__":
    test_multispecies_problem_initialisation()
