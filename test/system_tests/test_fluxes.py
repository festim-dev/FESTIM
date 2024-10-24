import numpy as np
import ufl
from dolfinx import fem

import festim as F

from .tools import error_L2

test_mesh_1d = F.Mesh1D(np.linspace(0, 1, 10000))
x_1d = ufl.SpatialCoordinate(test_mesh_1d.mesh)


def test_flux_bc_1_mobile_MMS_steady_state():
    """
    MMS test with a flux BC considering one mobile species at steady state
    """

    u_exact = lambda x: 1 + 2 * x[0] ** 2

    V = fem.functionspace(test_mesh_1d.mesh, ("Lagrange", 1))
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1
    T = 500
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = 500

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=u_exact, species=H),
        F.ParticleFluxBC(subdomain=right, value=4 * D, species=H),
    ]

    f = -ufl.div(D * ufl.grad(u_exact(x_1d)))
    my_model.sources = [F.ParticleSource(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, u_exact)

    assert L2_error < 1e-7


def test_flux_bc_heat_transfer_steady_state():
    """
    MMS test with a flux BC in a heat transfer problem at steady state
    """

    u_exact = lambda x: 1 + 2 * x[0] ** 2

    thermal_cond = 2.3

    my_model = F.HeatTransferProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=1, E_D=1, thermal_conductivity=thermal_cond)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    my_model.boundary_conditions = [
        F.FixedTemperatureBC(subdomain=left, value=u_exact),
        F.HeatFluxBC(subdomain=right, value=4 * thermal_cond),
    ]

    f = -ufl.div(thermal_cond * ufl.grad(u_exact(x_1d)))
    my_model.sources = [F.HeatSource(value=f, volume=vol)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    T_computed = my_model.u

    L2_error = error_L2(T_computed, u_exact)

    assert L2_error < 1e-7
