import numpy as np
import ufl
from dolfinx import fem

import festim as F

from .tools import error_L2

test_mesh_1d = F.Mesh1D(np.linspace(0, 1, 10000))
x_1d = ufl.SpatialCoordinate(test_mesh_1d.mesh)


def test_heat_transfer_steady_state():
    """
    MMS test with heat transfer simulation
    """

    def u_exact(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0])

    T_analytical_ufl = u_exact(ufl)
    T_analytical_np = u_exact(np)

    V = fem.functionspace(test_mesh_1d.mesh, ("Lagrange", 1))
    T_solution = fem.Function(V)
    T_solution.interpolate(lambda x: 1 + np.sin(2 * np.pi * x[0]))

    Thermal_conductivity = 1.5

    my_model = F.HeatTransferProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(
        name="mat", D_0=1, E_D=1, thermal_conductivity=Thermal_conductivity
    )
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    my_model.boundary_conditions = [
        F.FixedTemperatureBC(subdomain=left, value=T_analytical_ufl),
        F.FixedTemperatureBC(subdomain=right, value=T_analytical_ufl),
    ]

    f = -ufl.div(Thermal_conductivity * ufl.grad(T_analytical_ufl(x_1d)))
    my_model.sources = [F.HeatSource(value=f, volume=vol)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    T_computed = my_model.u

    L2_error = error_L2(T_computed, T_analytical_np)

    assert L2_error < 1e-7
