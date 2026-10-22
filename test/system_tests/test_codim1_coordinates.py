"""Radial transport coupled to manifolds, including point surfaces in 1D."""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F

from .tools import error_L2


@pytest.mark.parametrize(
    "coordinates,power", [("cartesian", 0), ("cylindrical", 1), ("spherical", 2)]
)
@pytest.mark.parametrize("interior", [False, True])
@pytest.mark.parametrize("transient", [False, True])
def test_radial_point_coupling(coordinates, power, interior, transient):
    """c_left=r+t, c_gamma=3+t, c_right=r+2+t exchange -1 and +1.

    The point at r=2 represents a cylindrical/spherical surface. Its storage and
    source must balance the bulk flux density, independently of physical area.
    """
    material = F.Material(D_0=1, E_D=0)
    left = F.VolumeSubdomain1D(id=1, borders=[1, 2], material=material)
    gamma = F.VolumeSubdomain(
        id=3, dim=0, material=material, locator=lambda x: np.isclose(x[0], 2)
    )
    inner = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1))
    bulk = F.Species("left", subdomains=[left])
    surface = F.Species("surface", subdomains=[gamma])
    subdomains = [left, gamma, inner]
    species = [bulk, surface]
    bcs = [F.FixedConcentrationBC(inner, value=lambda t: 1 + t, species=bulk)]
    sources = [
        F.ParticleSource(
            volume=left, value=lambda x: int(transient) - power / x[0], species=bulk
        ),
        F.ParticleSource(
            volume=gamma,
            value=lambda x: x[0] / 2 * (float(transient) + (0 if interior else 1)),
            species=surface,
        ),
    ]
    initial = []
    if transient:
        initial = [
            F.InitialConcentration(value=lambda x: x[0], volume=left, species=bulk),
            F.InitialConcentration(
                value=lambda x, T: (T - 499 + x[0] + 1) / 2,
                volume=gamma,
                species=surface,
            ),
        ]
    fluxes = [F.SurfaceFlux(field=bulk, surface=gamma)]
    if interior:
        right = F.VolumeSubdomain1D(id=2, borders=[2, 3], material=material)
        outer = F.SurfaceSubdomain(id=5, locator=lambda x: np.isclose(x[0], 3))
        other = F.Species("right", subdomains=[right])
        species.append(other)
        subdomains += [right, outer]
        bcs.append(F.FixedConcentrationBC(outer, value=lambda t: 5 + t, species=other))
        sources.append(
            F.ParticleSource(
                volume=right,
                value=lambda x: int(transient) - power / x[0],
                species=other,
            )
        )
        fluxes.append(F.SurfaceFlux(field=other, surface=gamma))
        if transient:
            initial.append(
                F.InitialConcentration(
                    value=lambda x: x[0] + 2, volume=right, species=other
                )
            )

    for sp in species:
        if sp is surface:
            continue
        dependencies = {"c_b": sp, "c_g": surface}
        bcs.append(
            F.ParticleFluxBC(
                gamma,
                species=sp,
                value=lambda c_b, c_g: c_g - c_b,
                species_dependent_value=dependencies,
            )
        )
        sources.append(
            F.ParticleSource(
                volume=gamma,
                species=surface,
                value=lambda c_b, c_g: c_b - c_g,
                species_dependent_value=dependencies,
            )
        )

    total = F.TotalVolume(field=surface, volume=gamma)
    average = F.AverageVolume(field=surface, volume=gamma)
    surface_total = F.TotalSurface(field=bulk, surface=gamma)
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh1D(
            np.linspace(1, 3 if interior else 2, 129), coordinate_system=coordinates
        ),
        subdomains=subdomains,
        species=species,
        sources=sources,
        boundary_conditions=bcs,
        initial_conditions=initial,
        temperature=lambda x: 500 + x[0],
        exports=[total, average, surface_total, *fluxes],
        settings=F.Settings(
            transient=transient,
            stepsize=F.Stepsize(0.1),
            final_time=0.2,
            atol=1e-11,
            rtol=1e-11,
        ),
    )
    model.show_progress_bar = False
    model.initialise()
    model.run()
    time = 0.2 if transient else 0
    area = {0: 1, 1: 4 * np.pi, 2: 16 * np.pi}[power]
    assert average.value == pytest.approx(3 + time, rel=2e-5)
    assert total.value == pytest.approx(area * (3 + time), rel=2e-5)
    assert surface_total.value == pytest.approx(area * (2 + time), rel=2e-5)
    assert fluxes[0].value == pytest.approx(-area, rel=2e-4)
    if interior:
        assert fluxes[1].value == pytest.approx(area, rel=2e-4)
    c = bulk.subdomain_to_post_processing_solution[left]
    assert error_L2(c, ufl.SpatialCoordinate(left.submesh)[0] + time) < 2e-5


@pytest.mark.parametrize("speed", [0.0, 0.6])
def test_cylindrical_manifold_radial_transport(speed):
    """A radial manifold: c_bulk=r²(1+z)+z², c_gamma=r²/2, J=r².

    The radius varies along Gamma, exposing a missing metric derivative in both
    tangential diffusion and drift. The ambient velocity's normal part is ignored.
    """
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([1.0, 0.0]), np.array([2.0, 1.0])], [32, 32]
    )
    bulk = F.VolumeSubdomain(id=1, material=F.Material(D_0=1, E_D=0))
    gamma = F.VolumeSubdomain(
        id=2,
        dim=1,
        material=F.Material(D_0=0.7, E_D=0),
        locator=lambda x: np.isclose(x[1], 0),
    )
    walls = F.SurfaceSubdomain(
        id=3,
        locator=lambda x: (
            np.isclose(x[0], 1) | np.isclose(x[0], 2) | np.isclose(x[1], 1)
        ),
    )
    ends = F.SurfaceSubdomain(
        id=4, dim=0, locator=lambda x: np.isclose(x[0], 1) | np.isclose(x[0], 2)
    )
    outlet = F.SurfaceSubdomain(id=5, dim=0, locator=lambda x: np.isclose(x[0], 2))
    c_b = F.Species("bulk", subdomains=[bulk])
    c_g = F.Species("manifold", subdomains=[gamma])
    dependencies = {"c_b": c_b, "c_g": c_g}
    total = F.TotalVolume(c_g, gamma)
    average = F.AverageVolume(c_g, gamma)
    surface_total = F.TotalSurface(c_b, gamma)
    surface_average = F.AverageSurface(c_b, gamma)
    exchange = F.SurfaceFlux(c_b, gamma)
    outlet_flux = F.SurfaceFlux(c_g, outlet)
    velocity = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1, (2,))))
    velocity.interpolate(lambda x: np.tile([speed, 0.8], (x.shape[1], 1)).T)
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh, coordinate_system="cylindrical"),
        subdomains=[bulk, gamma, walls, ends, outlet],
        species=[c_b, c_g],
        temperature=500,
        settings=F.Settings(transient=False, atol=1e-11, rtol=1e-11),
        boundary_conditions=[
            F.FixedConcentrationBC(
                walls, value=lambda x: x[0] ** 2 * (1 + x[1]) + x[1] ** 2, species=c_b
            ),
            F.FixedConcentrationBC(ends, value=lambda x: 0.5 * x[0] ** 2, species=c_g),
            F.ParticleFluxBC(
                gamma,
                value=lambda c_b, c_g: 2 * (c_g - c_b),
                species=c_b,
                species_dependent_value=dependencies,
            ),
        ],
        sources=[
            F.ParticleSource(volume=bulk, species=c_b, value=lambda x: -6 - 4 * x[1]),
            F.ParticleSource(
                volume=gamma,
                species=c_g,
                value=lambda x: -1.4 - x[0] ** 2 + 1.5 * speed * x[0],
            ),
            F.ParticleSource(
                volume=gamma,
                species=c_g,
                value=lambda c_b, c_g: 2 * (c_b - c_g),
                species_dependent_value=dependencies,
            ),
        ],
        drift_terms=[
            F.AdvectionTerm(
                velocity=velocity,
                subdomain=gamma,
                species=c_g,
            )
        ]
        if speed
        else [],
        exports=[total, average, surface_total, surface_average, exchange, outlet_flux],
    )
    model.initialise()
    model.run()
    x_b = ufl.SpatialCoordinate(bulk.submesh)
    x_g = ufl.SpatialCoordinate(gamma.submesh)
    assert (
        error_L2(
            c_b.subdomain_to_post_processing_solution[bulk],
            x_b[0] ** 2 * (1 + x_b[1]) + x_b[1] ** 2,
        )
        < 1e-3
    )
    assert (
        error_L2(c_g.subdomain_to_post_processing_solution[gamma], 0.5 * x_g[0] ** 2)
        < 1e-3
    )
    assert total.value == pytest.approx(15 * np.pi / 4, rel=1e-3)
    assert average.value == pytest.approx(1.25, rel=1e-3)
    assert surface_total.value == pytest.approx(15 * np.pi / 2, rel=1e-3)
    assert surface_average.value == pytest.approx(2.5, rel=1e-3)
    # Fluxes use the P1 gradient, which is only first-order accurate.
    assert exchange.value == pytest.approx(15 * np.pi / 2, rel=0.04)
    assert outlet_flux.value == pytest.approx(4 * np.pi * (2 * speed - 1.4), abs=0.2)


def test_spherical_point_trapping():
    """Two independent point surfaces retain their own coordinates and inventories.

    With n_traps=c_initial=r and irreversible trapping, c_mobile'=-k*c_mobile².
    Compare each backward-Euler step with the positive root of that quadratic.
    """
    material = F.Material(D_0=1, E_D=0)
    bulk = F.VolumeSubdomain1D(id=1, borders=[1, 3], material=material)
    gamma = F.VolumeSubdomain(
        id=2,
        dim=0,
        material=material,
        locator=lambda x: np.isclose(x[0], 1.5) | np.isclose(x[0], 2.5),
    )
    bulk_species = F.Species("bulk", subdomains=[bulk])
    mobile = F.Species("mobile", subdomains=[gamma])
    trapped = F.Species("trapped", mobile=False, subdomains=[gamma])
    empty = F.ImplicitSpecies(n=lambda x: x[0], others=[trapped])
    total_mobile = F.TotalVolume(mobile, gamma)
    total_trapped = F.TotalVolume(trapped, gamma)
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh1D(np.linspace(1, 3, 17), coordinate_system="spherical"),
        subdomains=[bulk, gamma],
        species=[bulk_species, mobile, trapped],
        initial_conditions=[
            F.InitialConcentration(value=lambda x: x[0], volume=gamma, species=mobile)
        ],
        reactions=[
            F.GenericReaction(
                reactant=[mobile, empty],
                product=trapped,
                volume=gamma,
                forward_rate=lambda x, t, T: x[0] * (1 + t) / T,
            )
        ],
        temperature=lambda x, t: 500 + x[0] + t,
        settings=F.Settings(
            transient=True,
            stepsize=F.Stepsize(0.1),
            final_time=0.2,
            atol=1e-12,
            rtol=1e-12,
        ),
        exports=[total_mobile, total_trapped],
    )
    model.show_progress_bar = False
    model.initialise()
    model.run()
    radii = np.array([1.5, 2.5])
    expected = radii.copy()
    for time in (0.1, 0.2):
        coefficient = 0.1 * radii * (1 + time) / (500 + radii + time)
        expected = 2 * expected / (1 + np.sqrt(1 + 4 * coefficient * expected))
    weights = 4 * np.pi * radii**2
    assert total_mobile.value == pytest.approx(np.dot(weights, expected), rel=1e-10)
    assert total_trapped.value == pytest.approx(
        np.dot(weights, radii - expected), rel=1e-9
    )
    assert total_mobile.value + total_trapped.value == pytest.approx(
        np.dot(weights, radii), rel=1e-12
    )
