import festim as F
import numpy as np


def test():
    my_problem = F.HeatTransferProblem()

    my_problem.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 100))
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)
    my_problem.surface_subdomains = [left, right]
    mat = F.Material(D_0=None, E_D=None)
    mat.thermal_conductivity = 2.0

    my_problem.volume_subdomains = [
        F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
    ]

    my_problem.boundary_conditions = [
        F.FixedTemperatureBC(subdomain=left, value=0),
        F.FixedTemperatureBC(subdomain=right, value=0),
    ]

    my_problem.sources = [
        F.HeatSource(value=1, volume=my_problem.volume_subdomains[0])
    ]

    my_problem.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=False,
    )

    my_problem.initialise()
    my_problem.run()
