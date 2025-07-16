import numpy as np
from dolfinx import fem

import festim as F

from .tools import error_L2


def test_run_MMS_spherical():
    """
    Tests that festim produces the correct concentration field in spherical
    coordinates
    """

    my_mesh = F.Mesh1D(vertices=np.linspace(1, 2, 1000), coordinate_system="spherical")
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))

    u_exact = lambda x: 3 + 2 * x[0] ** 2

    f = -12

    my_mat = F.Material(D_0=1.0, E_D=0)

    left = F.SurfaceSubdomain1D(id=1, x=1)
    right = F.SurfaceSubdomain1D(id=2, x=2)
    my_vol = F.VolumeSubdomain1D(id=3, borders=[1, 2], material=my_mat)

    my_subdomains = [my_vol, left, right]

    H = F.Species("H")

    my_bcs = [
        F.FixedConcentrationBC(subdomain=left, value=u_exact, species=H),
        F.FixedConcentrationBC(subdomain=right, value=u_exact, species=H),
    ]

    my_temp = 500

    my_sources = [
        F.ParticleSource(value=f, volume=my_vol, species=H),
    ]

    my_settings = F.Settings(
        atol=1e-10,
        rtol=1e-9,
        max_iterations=50,
        transient=False,
    )

    my_sim = F.HydrogenTransportProblem(
        mesh=my_mesh,
        species=[H],
        subdomains=my_subdomains,
        boundary_conditions=my_bcs,
        temperature=my_temp,
        sources=my_sources,
        settings=my_settings,
    )

    my_sim.initialise()
    my_sim.run()

    computed_solution = H.post_processing_solution

    L2_error = error_L2(computed_solution, u_exact)

    assert L2_error < 1e-6
