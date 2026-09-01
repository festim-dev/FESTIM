import math

import numpy as np
from dolfinx import fem

import festim as F

from .tools import error_L2


def test_run_MMS_spherical():
    """Tests that festim produces the correct concentration field in spherical
    coordinates."""

    my_mesh = F.Mesh1D(vertices=np.linspace(1, 2, 1000), coordinate_system="spherical")
    fem.functionspace(my_mesh.mesh, ("Lagrange", 1))

    def u_exact(x):
        return 3 + 2 * x[0] ** 2

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


def test_surface_flux_spherical():
    """Tests that SurfaceFlux computes the correct flux in spherical coordinates.

    Uses the analytical solution for steady-state diffusion through a spherical
    shell with no source: u(r) = A/r + B, solved for fixed concentrations u1, u2
    at the inner and outer radii r1, r2. The total flux through any spherical
    shell, Q = 4 * pi * D * A, is constant in r (steady-state conservation with no
    source), so the inner and outer surface fluxes should be equal and opposite.
    """

    r1, r2 = 1.0, 2.0
    c1, c2 = 10.0, 2.0
    D = 2.0

    my_mesh = F.Mesh1D(
        vertices=np.linspace(r1, r2, 1000), coordinate_system="spherical"
    )

    my_mat = F.Material(D_0=D, E_D=0)

    left = F.SurfaceSubdomain1D(id=1, x=r1)
    right = F.SurfaceSubdomain1D(id=2, x=r2)
    my_vol = F.VolumeSubdomain1D(id=3, borders=[r1, r2], material=my_mat)

    H = F.Species("H")

    my_bcs = [
        F.FixedConcentrationBC(subdomain=left, value=c1, species=H),
        F.FixedConcentrationBC(subdomain=right, value=c2, species=H),
    ]

    flux_left = F.SurfaceFlux(field=H, surface=left)
    flux_right = F.SurfaceFlux(field=H, surface=right)

    my_settings = F.Settings(
        atol=1e-10,
        rtol=1e-9,
        max_iterations=50,
        transient=False,
    )

    my_sim = F.HydrogenTransportProblem(
        mesh=my_mesh,
        species=[H],
        subdomains=[my_vol, left, right],
        boundary_conditions=my_bcs,
        temperature=500,
        exports=[flux_left, flux_right],
        settings=my_settings,
    )

    my_sim.initialise()
    my_sim.run()

    A = (c1 - c2) * r1 * r2 / (r2 - r1)
    expected_flux_magnitude = 4 * math.pi * D * A

    assert np.isclose(flux_left.value, -expected_flux_magnitude, rtol=1e-2)
    assert np.isclose(flux_right.value, expected_flux_magnitude, rtol=1e-2)


def test_run_MMS_spherical_mixed_domain():
    """Tests that festim produces the correct concentration field in spherical
    coordinates in a discontinuous domain with two materials."""

    my_model = F.HydrogenTransportProblemDiscontinuous()

    r_interface = 1.5
    left_domain = np.linspace(1, r_interface, num=1000)
    right_domain = np.linspace(r_interface, r_interface + 1, num=1000)

    vertices = np.concatenate(
        [
            left_domain,
            right_domain,
        ]
    )
    my_mesh = F.Mesh1D(vertices=vertices, coordinate_system="spherical")

    my_model.mesh = my_mesh

    K_S_left = 2.0
    K_S_right = 3.0
    D = 2.0

    def c_exact_left(x):
        return (r_interface - x[0]) ** 2 + 1.2

    def c_exact_right(x):
        return K_S_right / K_S_left * c_exact_left(x)

    def lap_c(r):
        return -(4 * r_interface / r) + 6

    mat_1 = F.Material(D_0=D, E_D=0, K_S_0=K_S_left, E_K_S=0, solubility_law="sievert")
    mat_2 = F.Material(D_0=D, E_D=0, K_S_0=K_S_right, E_K_S=0, solubility_law="sievert")

    left = F.SurfaceSubdomain1D(id=1, x=left_domain[0])
    right = F.SurfaceSubdomain1D(id=2, x=right_domain[-1])
    vol_1 = F.VolumeSubdomain1D(
        id=3, borders=[left_domain[0], left_domain[-1]], material=mat_1
    )
    vol_2 = F.VolumeSubdomain1D(
        id=4, borders=[right_domain[0], right_domain[-1]], material=mat_2
    )

    my_model.subdomains = [vol_1, vol_2, left, right]

    my_model.interfaces = [F.Interface(5, (vol_1, vol_2), penalty_term=1e5)]

    H = F.Species("H", mobile=True, subdomains=[vol_1, vol_2])
    my_model.species = [H]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left, value=c_exact_left, species=H),
        F.FixedConcentrationBC(subdomain=right, value=c_exact_right, species=H),
    ]

    my_model.temperature = 500

    def f_left(x):
        return -D * lap_c(x[0])

    def f_right(x):
        return -D * K_S_right / K_S_left * lap_c(x[0])

    my_model.sources = [
        F.ParticleSource(value=f_left, volume=vol_1, species=H),
        F.ParticleSource(value=f_right, volume=vol_2, species=H),
    ]

    my_model.settings = F.Settings(
        atol=1e-10, rtol=1e-10, max_iterations=10, transient=False, element_degree=2
    )

    my_model.initialise()
    my_model.run()

    c_l_computed = H.subdomain_to_post_processing_solution[vol_1]
    c_r_computed = H.subdomain_to_post_processing_solution[vol_2]

    L2_error_l = error_L2(c_l_computed, c_exact_left)
    L2_error_r = error_L2(c_r_computed, c_exact_right)

    assert L2_error_l < 1e-06
    assert L2_error_r < 1e-06
