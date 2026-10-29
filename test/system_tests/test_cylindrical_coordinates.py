import math

from mpi4py import MPI

import dolfinx
import numpy as np
import ufl

import festim as F

from .tools import error_L2


def test_run_MMS_cylindrical():
    """Tests that festim produces the correct concentration field in cylindrical
    coordinates."""

    my_mesh = F.Mesh1D(vertices=np.linspace(1, 2, 500), coordinate_system="cylindrical")

    def u_exact(x):
        return 1 + x[0] ** 2

    f = -4

    my_mat = F.Material(D_0=1.0, E_D=0)

    left = F.SurfaceSubdomain1D(id=1, x=1)
    right = F.SurfaceSubdomain1D(id=2, x=2)
    my_vol = F.VolumeSubdomain1D(id=3, borders=[1, 2], material=my_mat)

    my_subdomains = [my_vol, left, right]

    H = F.Species("H")
    D = F.Species("D")

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
        species=[H, D],
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


def test_surface_flux_cylindrical():
    """Tests that SurfaceFlux computes the correct flux in cylindrical coordinates,
    on a 1D radial mesh.

    Uses the analytical solution for steady-state diffusion through a cylindrical
    shell with no source: u(r) = (C1/D) * ln(r) + C2, solved for fixed
    concentrations c1, c2 at the inner and outer radii r1, r2. The total flux
    (per unit axial length) through any cylindrical shell, Q = -2 * pi * C1, is
    constant in r (steady-state conservation with no source), so the inner and
    outer surface fluxes should be equal and opposite.
    """

    r1, r2 = 1.0, 2.0
    c1, c2 = 10.0, 2.0
    D = 2.0

    my_mesh = F.Mesh1D(
        vertices=np.linspace(r1, r2, 1000), coordinate_system="cylindrical"
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

    C1 = D * (c1 - c2) / math.log(r1 / r2)
    expected_flux_magnitude = abs(2 * math.pi * C1)

    assert np.isclose(flux_left.value, -expected_flux_magnitude, rtol=1e-2)
    assert np.isclose(flux_right.value, expected_flux_magnitude, rtol=1e-2)


def test_run_MMS_cylindrical_mixed_domain():
    """Tests that festim produces the correct concentration field in cylindrical
    coordinates in a discontinuous domain with two materials."""

    my_model = F.HydrogenTransportProblemDiscontinuous()

    r_interface = 2
    left_domain = np.linspace(1, r_interface, num=1000)
    right_domain = np.linspace(r_interface, r_interface + 1, num=1000)

    vertices = np.concatenate(
        [
            left_domain,
            right_domain,
        ]
    )
    my_mesh = F.Mesh1D(vertices=vertices, coordinate_system="cylindrical")

    my_model.mesh = my_mesh

    K_S_left = 3.0
    K_S_right = 2.0
    D = 2.0

    def c_exact_left(x):
        return (r_interface - x[0]) ** 2 + 2

    def c_exact_right(x):
        return K_S_right / K_S_left * c_exact_left(x)

    def lap_c(r):
        return 4 - 2 * r_interface / r

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

    my_model.interfaces = [F.Interface(5, (vol_1, vol_2), penalty_term=100)]

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
        atol=1e-10, rtol=1e-10, max_iterations=10, transient=False, element_degree=1
    )

    my_model.initialise()
    my_model.run()

    c_l_computed = H.subdomain_to_post_processing_solution[vol_1]
    c_r_computed = H.subdomain_to_post_processing_solution[vol_2]

    L2_error_l = error_L2(c_l_computed, c_exact_left)
    L2_error_r = error_L2(c_r_computed, c_exact_right)

    assert L2_error_l < 1e-06
    assert L2_error_r < 1e-06


def test_surface_flux_cylindrical_2d_oblique_surface():
    """Tests that SurfaceFlux is correct in cylindrical coordinates on a 2D (r, z)
    mesh whose boundary is not aligned with the mesh axes.

    The domain is the meridian section of a truncated cone: r goes from ``r1`` to
    ``r2`` and z from 0 to the sloped line ``h(r) = z0 + slope * (r - r1)``. The
    sloped boundary is the lateral surface of the cone, so its normal has both an r
    and a z component -- unlike the inner, outer and bottom surfaces, which are
    axis-aligned and are checked here too.

    The concentration is fixed to the manufactured solution ``c = 1 + r**2 + z**2``
    on every boundary, with the matching source term, so the flux density is
    ``q = -D grad(c) = -2 D (r, z)``. Integrating ``q . n`` over a surface of
    revolution uses ``dA = 2 pi r dl``, which on the sloped boundary gives

        Q_cone = 2 pi D (slope * r1 - z0) * (r2**2 - r1**2)

    since the integrand ``r * (slope * r - h(r))`` collapses to the constant
    ``r * (slope * r1 - z0)``. The axis-aligned surfaces integrate to
    ``4 pi D r1**2 z0`` (inner) and ``-4 pi D r2**2 h(r2)`` (outer), and the bottom
    carries no flux because ``q . n = -q_z`` vanishes at z = 0. The four fluxes are
    also checked against the divergence theorem, which ties the weight used on the
    sloped surface to the one used on the axis-aligned ones.
    """

    r1, r2 = 1.0, 2.0
    z0, slope = 1.0, 0.5
    D = 2.0

    def h(r):
        """z of the sloped boundary at radius r."""
        return z0 + slope * (r - r1)

    # deform a unit square into the (r, z) meridian section of the truncated cone.
    # The mapping is affine in r, so the sloped boundary stays a straight line and the
    # straight-sided cells represent the geometry exactly
    square = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 24, 24)
    nodes = square.geometry.x
    r = r1 + (r2 - r1) * nodes[:, 0].copy()
    nodes[:, 0] = r
    nodes[:, 1] = nodes[:, 1].copy() * h(r)

    my_mesh = F.Mesh(mesh=square, coordinate_system="cylindrical")

    def c_exact(x):
        return 1 + x[0] ** 2 + x[1] ** 2

    my_mat = F.Material(D_0=D, E_D=0)

    my_vol = F.VolumeSubdomain(
        id=1,
        material=my_mat,
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    cone = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], h(x[0])))
    bottom = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[1], 0.0))
    inner = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], r1))
    outer = F.SurfaceSubdomain(id=5, locator=lambda x: np.isclose(x[0], r2))
    surfaces = [cone, bottom, inner, outer]

    H = F.Species("H")

    # the cylindrical divergence of the flux, ie. the source that makes c_exact the
    # solution: (1 / r) d/dr (r q_r) + d q_z / dz
    x = ufl.SpatialCoordinate(my_mesh.mesh)
    f = -(1 / x[0]) * ufl.div(x[0] * D * ufl.grad(c_exact(x)))

    fluxes = {surf.id: F.SurfaceFlux(field=H, surface=surf) for surf in surfaces}

    my_sim = F.HydrogenTransportProblem(
        mesh=my_mesh,
        species=[H],
        subdomains=[my_vol, *surfaces],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=surf, value=c_exact, species=H)
            for surf in surfaces
        ],
        temperature=500,
        sources=[F.ParticleSource(value=f, volume=my_vol, species=H)],
        exports=list(fluxes.values()),
        # c_exact is quadratic and the cells are straight-sided, so P2 reproduces it
        # exactly and the fluxes come out to solver precision
        settings=F.Settings(
            atol=1e-12, rtol=1e-12, max_iterations=50, transient=False, element_degree=2
        ),
    )

    my_sim.initialise()
    my_sim.run()

    expected_cone = 2 * math.pi * D * (slope * r1 - z0) * (r2**2 - r1**2)
    expected_inner = 4 * math.pi * D * r1**2 * z0
    expected_outer = -4 * math.pi * D * r2**2 * h(r2)

    assert np.isclose(fluxes[cone.id].value, expected_cone, rtol=1e-6)
    assert np.isclose(fluxes[inner.id].value, expected_inner, rtol=1e-6)
    assert np.isclose(fluxes[outer.id].value, expected_outer, rtol=1e-6)
    assert np.isclose(fluxes[bottom.id].value, 0.0, atol=1e-6)

    # divergence theorem: the net flux out of the closed boundary is the integral of
    # div(q) = f = -6 D over the volume of revolution, 2 pi times the integral of
    # r h(r) dr
    volume = (
        2
        * math.pi
        * ((z0 - slope * r1) * (r2**2 - r1**2) / 2 + slope * (r2**3 - r1**3) / 3)
    )
    net_flux = sum(flux.value for flux in fluxes.values())

    assert np.isclose(net_flux, -6 * D * volume, rtol=1e-6)
