"""Derived quantities on and around a codim-1 (manifold) subdomain. See issue #1208.

Three places a quantity can be asked for once a manifold is in the mesh, none of which
the export layer could reach before:

- the **facets a manifold occupies**, for a *bulk* species -- the exchange between the
  bulk and the manifold. On an interior manifold this needs the ``dS`` coupling measure
  and the restriction of the side the species lives on: the parent ``ds`` integrates to
  exactly zero over interior facets, so getting this wrong is a silent zero rather than
  an error;
- the **manifold itself**, for its own species, integrated over its submesh;
- the **boundary of a manifold**, which is codim-1 *relative to the manifold* and so an
  ordinary exterior facet integral on its submesh -- the outlet of the pipe of
  ``test_codim1_boundary.py``.

Every value asserted here is a closed form, and the sign convention is the one an
ordinary surface flux already has: positive means leaving the subdomain the species
lives on.
"""

from itertools import pairwise

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F

D_O, D_G, K_EX = 1.5, 0.7, 2.0
BETA = 1.0 - D_O / K_EX
OMEGA_ID, GAMMA_ID, RIGHT_ID, END_ID = 1, 2, 3, 4

pi = np.pi


def exterior_model(n=32, exports=()):
    """The manufactured problem of ``test_codim1_coupling.py`` in 2D, with Gamma on the
    outer boundary ``x=0``::

        c_O = 1 + x^2 + (1 + x) cos(pi y)      c_G = 1 + beta cos(pi y)

    so the quantities below are all known in closed form.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)

    omega = F.VolumeSubdomain(
        id=OMEGA_ID,
        material=F.Material(D_0=D_O, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    gamma = F.VolumeSubdomain(
        id=GAMMA_ID,
        material=F.Material(D_0=D_G, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], 0.0),
    )
    right = F.SurfaceSubdomain(id=RIGHT_ID, locator=lambda x: np.isclose(x[0], 1.0))

    H_om = F.Species("H_om", subdomains=[omega])
    H_gam = F.Species("H_gam", subdomains=[gamma])

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_om, H_gam],
        subdomains=[omega, gamma, right],
        sources=[
            F.ParticleSource(
                value=lambda x: -D_O * (2 - pi**2 * (1 + x[0]) * ufl.cos(pi * x[1])),
                species=H_om,
                volume=omega,
            ),
            F.ParticleSource(
                value=lambda x: (D_G * pi**2 * BETA - D_O) * ufl.cos(pi * x[1]),
                species=H_gam,
                volume=gamma,
            ),
            F.ParticleSource(
                value=lambda c_g, c_o: K_EX * (c_o - c_g),
                species=H_gam,
                volume=gamma,
                species_dependent_value={"c_o": H_om, "c_g": H_gam},
            ),
        ],
        boundary_conditions=[
            F.ParticleFluxBC(
                subdomain=gamma,
                value=lambda c_g, c_o: K_EX * (c_g - c_o),
                species=H_om,
                species_dependent_value={"c_o": H_om, "c_g": H_gam},
            ),
            F.FixedConcentrationBC(
                subdomain=right,
                value=lambda x: 1 + x[0] ** 2 + (1 + x[0]) * ufl.cos(pi * x[1]),
                species=H_om,
            ),
        ],
        exports=list(exports),
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    return model, (omega, gamma, right), (H_om, H_gam)


def by_title(model):
    return {e.title: e.data[-1] for e in model.exports}


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_quantities_on_an_exterior_manifold():
    """Bulk quantities on the facets of a manifold, and manifold quantities over it.

    Exact values, with Gamma the unit-length segment ``x=0``::

        int_Gamma c_O  = int_0^1 (1 + cos(pi y)) dy = 1
        int_Gamma c_G  = int_0^1 (1 + beta cos(pi y)) dy = 1
    """
    model, (_omega, gamma, _right), (H_om, H_gam) = exterior_model()
    model.exports = [
        F.TotalSurface(field=H_om, surface=gamma),
        F.AverageSurface(field=H_om, surface=gamma),
        F.TotalVolume(field=H_gam, volume=gamma),
        F.AverageVolume(field=H_gam, volume=gamma),
    ]
    model.initialise()
    model.run()

    got = by_title(model)
    assert np.isclose(got["Total H_om surface 2"], 1.0, atol=1e-6)
    assert np.isclose(got["Average H_om surface 2"], 1.0, atol=1e-6)
    assert np.isclose(got["Total H_gam volume 2"], 1.0, atol=1e-6)
    assert np.isclose(got["Average H_gam volume 2"], 1.0, atol=1e-6)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_flux_on_an_exterior_manifold_matches_the_exchange():
    """The bulk flux on Gamma's facets is the exchange term that drives the coupling.

    ``-D_O grad(c_O).n = J = K_EX (c_O - c_G) = D_O cos(pi y)`` there, which integrates
    to zero over Gamma -- so the total alone would pass with almost any sign or measure.
    The flux is therefore also checked pointwise against ``J``, by comparing it with the
    exchange assembled independently from the two solutions, and by refining.
    """
    errors = []
    for n in (16, 32, 64):
        model, (omega, gamma, _right), (H_om, _H_gam) = exterior_model(n=n)
        model.exports = [F.SurfaceFlux(field=H_om, surface=gamma)]
        model.initialise()
        model.run()

        flux = model.exports[0].data[-1]
        assert np.isclose(flux, 0.0, atol=1e-3)

        # the same quantity weighted by cos(pi y), whose exact value is
        # int_0^1 D_O cos^2(pi y) dy = D_O / 2, is not degenerate
        c_om = H_om.subdomain_to_post_processing_solution[omega]
        parent = model.mesh.mesh
        y = ufl.SpatialCoordinate(parent)[1]
        weighted = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(
                -D_O
                * ufl.dot(ufl.grad(c_om), ufl.FacetNormal(parent))
                * ufl.cos(pi * y)
                * model.ds(gamma.id),
                entity_maps=[sd.cell_map for sd in model.volume_subdomains],
            )
        )
        errors.append(abs(weighted - D_O / 2))

    rates = [np.log(e0 / e1) / np.log(2) for e0, e1 in pairwise(errors)]
    assert all(r > 1.8 for r in rates), rates


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_flux_on_an_interior_manifold_is_not_silently_zero():
    """The case the parent ``ds`` gets wrong without failing.

    Steady 1D transport across a manifold at ``x=0.5``, ``c=2`` at ``x=0`` and ``c=0``
    at ``x=1`` (the configuration of ``test_codim1_interface.py``). Every flux equals
    the same ``J`` at steady state::

        D/0.5 (2 - c_L) = K_L (c_L - c_G) = K_R (c_G - c_R) = D/0.5 c_R = J

    giving ``J = 4/3``, ``c_L = 14/9``, ``c_G = 8/9``, ``c_R = 4/9``. Gamma has unit
    length, so the integrated flux is ``J`` itself -- positive leaving the left bulk,
    negative on the right, which is entering it. The exchange rates differ on the two
    sides, so a swapped ``"+"``/``"-"`` would give the wrong number rather than the
    same one twice.
    """
    from .test_codim1_interface import D_BULK, build

    model, (left, right, gamma), (H_l, H_r, H_g) = build(n=20)
    model.exports = [
        F.SurfaceFlux(field=H_l, surface=gamma),
        F.SurfaceFlux(field=H_r, surface=gamma),
        F.TotalSurface(field=H_l, surface=gamma),
        F.TotalSurface(field=H_r, surface=gamma),
        F.AverageSurface(field=H_l, surface=gamma),
        F.AverageSurface(field=H_r, surface=gamma),
        F.TotalVolume(field=H_g, volume=gamma),
    ]
    model.initialise()
    model.run()

    got = by_title(model)
    # not merely nonzero: the parent ds would give exactly 0.0 for both
    assert np.isclose(got["H_l flux surface 3"], 4 / 3, atol=1e-10)
    assert np.isclose(got["H_r flux surface 3"], -4 / 3, atol=1e-10)
    assert np.isclose(got["Total H_l surface 3"], 14 / 9, atol=1e-10)
    assert np.isclose(got["Total H_r surface 3"], 4 / 9, atol=1e-10)
    assert np.isclose(got["Total H_g volume 3"], 8 / 9, atol=1e-10)
    # the solution is uniform along Gamma and Gamma has unit measure, so the averages
    # are the totals -- which also pins the measure the average divides by
    assert np.isclose(got["Average H_l surface 3"], 14 / 9, atol=1e-10)
    assert np.isclose(got["Average H_r surface 3"], 4 / 9, atol=1e-10)

    # the same two fluxes assembled by hand from the solutions, on the coupling measure
    # and restriction the *formulation* uses, so they go nowhere near the export layer:
    # a wrong integrand, measure or side in the exports shows up as a disagreement here
    # even if the closed form above were ever relaxed
    parent = model.mesh.mesh
    n = ufl.FacetNormal(parent)
    dS_gamma = model.facet_measure(gamma)(gamma.id)
    for species, volume, title in (
        (H_l, left, "H_l flux surface 3"),
        (H_r, right, "H_r flux surface 3"),
    ):
        c = species.subdomain_to_post_processing_solution[volume]
        by_hand = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(
                model.restrict(
                    -D_BULK * ufl.dot(ufl.grad(c), n),
                    model.restriction_of(gamma, volume),
                )
                * dS_gamma,
                entity_maps=[sd.cell_map for sd in model.volume_subdomains],
            )
        )
        assert np.isclose(by_hand, got[title], atol=1e-10)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_interior_manifold_flux_balance():
    """What the manifold takes from one side it gives to the other, at steady state
    with no source on it. This is the check that catches a restriction applied to only
    part of the integrand, which a convergence study can miss."""
    from .test_codim1_interface import build

    model, (_left, _right, gamma), (H_l, H_r, _H_g) = build(n=20)
    model.exports = [
        F.SurfaceFlux(field=H_l, surface=gamma),
        F.SurfaceFlux(field=H_r, surface=gamma),
    ]
    model.initialise()
    model.run()

    got = by_title(model)
    into_gamma, out_of_gamma = got["H_l flux surface 3"], got["H_r flux surface 3"]
    assert np.isclose(into_gamma + out_of_gamma, 0.0, atol=1e-10)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_quantities_on_the_boundary_of_a_manifold():
    """Gamma's own endpoint: an exterior facet integral on Gamma's submesh.

    At ``y=1`` the manufactured manifold solution gives ``c_G = 1 + beta cos(pi) =
    1 - beta = 0.75``, and Gamma's endpoint is a single point of unit measure, so the
    total is that value. Its natural condition ``dc_G/dy = 0`` holds exactly there, so
    the flux converges to zero -- at first order, being a derivative recovered at a
    boundary point.
    """
    values, fluxes = [], []
    for n in (16, 32, 64):
        model, (_omega, _gamma, _right), (_H_om, H_gam) = exterior_model(n=n)
        end = F.SurfaceSubdomain(
            id=END_ID, dim=0, locator=lambda x: np.isclose(x[1], 1.0)
        )
        model.exports = [
            F.TotalSurface(field=H_gam, surface=end),
            F.SurfaceFlux(field=H_gam, surface=end),
        ]
        model.initialise()
        model.run()

        got = by_title(model)
        values.append(abs(got[f"Total H_gam surface {END_ID}"] - 0.75))
        fluxes.append(abs(got[f"H_gam flux surface {END_ID}"]))

    assert values[-1] < 1e-4, values
    # both converge; the flux only at first order
    assert all(v1 < v0 for v0, v1 in pairwise(values)), values
    assert all(f1 < 0.6 * f0 for f0, f1 in pairwise(fluxes)), fluxes


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_outlet_flux_of_a_manifold_carrying_advection():
    """A manifold's endpoint flux against a closed form that is not zero.

    Gamma alone, uncoupled from the bulk, with advection-diffusion along it and
    Dirichlet at both ends (the two-point BVP of ``test_codim1_boundary.py``)::

        c = (e^Pe - e^(v y / D)) / (e^Pe - 1),   Pe = v L / D

    so the diffusive flux at the outlet ``y=1`` is
    ``-D dc/dy = v e^Pe / (e^Pe - 1)``, and the export must reproduce it. This is the
    quantity a pipe model asks for: what leaves the fluid at the outlet.
    """
    v_y, D_gam, n = 2.0, 0.7, 200
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, n)

    omega = F.VolumeSubdomain(
        id=OMEGA_ID,
        material=F.Material(D_0=1.0, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    gamma = F.VolumeSubdomain(
        id=GAMMA_ID,
        material=F.Material(D_0=D_gam, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], 0.0),
    )
    inlet = F.SurfaceSubdomain(id=3, dim=0, locator=lambda x: np.isclose(x[1], 0.0))
    outlet = F.SurfaceSubdomain(id=4, dim=0, locator=lambda x: np.isclose(x[1], 1.0))
    fixed = F.SurfaceSubdomain(id=5, locator=lambda x: np.isclose(x[0], 1.0))

    H_om = F.Species("H_om", subdomains=[omega])
    H_gam = F.Species("H_gam", subdomains=[gamma])

    vel = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,))))
    vel.interpolate(lambda x: np.vstack([np.zeros_like(x[0]), np.full_like(x[0], v_y)]))

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_om, H_gam],
        subdomains=[omega, gamma, inlet, outlet, fixed],
        drift_terms=[F.AdvectionTerm(velocity=vel, subdomain=gamma, species=H_gam)],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=inlet, value=1.0, species=H_gam),
            F.FixedConcentrationBC(subdomain=outlet, value=0.0, species=H_gam),
            F.FixedConcentrationBC(subdomain=fixed, value=0.0, species=H_om),
        ],
        exports=[F.SurfaceFlux(field=H_gam, surface=outlet)],
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    model.initialise()
    model.run()

    peclet = v_y / D_gam
    expected = v_y * np.exp(peclet) / (np.exp(peclet) - 1)
    assert np.isclose(model.exports[0].data[-1], expected, rtol=2e-2), (
        model.exports[0].data[-1],
        expected,
    )


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_manifolds_own_species_on_its_own_facets_raises():
    """A manifold's species has no flux *across* the manifold -- the quantity the user
    means is a volume one over it."""
    model, (_omega, gamma, _right), (_H_om, H_gam) = exterior_model(n=8)
    model.exports = [F.SurfaceFlux(field=H_gam, surface=gamma)]

    with pytest.raises(ValueError, match="has no flux across it"):
        model.initialise()


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_export_on_a_manifold_boundary_locator_matching_nothing_raises():
    """A locator selecting a point interior to Gamma would otherwise report zero."""
    model, (_omega, _gamma, _right), (_H_om, H_gam) = exterior_model(n=8)
    inside = F.SurfaceSubdomain(
        id=END_ID, dim=0, locator=lambda x: np.isclose(x[1], 0.5)
    )
    model.exports = [F.TotalSurface(field=H_gam, surface=inside)]

    with pytest.raises(ValueError, match="matched no boundary entity"):
        model.initialise()


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_ordinary_volume_subdomain_as_a_surface_raises():
    """Only a codim-1 volume subdomain occupies facets."""
    model, (omega, _gamma, _right), (H_om, _H_gam) = exterior_model(n=8)
    model.exports = [F.SurfaceFlux(field=H_om, surface=omega)]

    with pytest.raises(ValueError, match="is not a manifold"):
        model.initialise()


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_volume_as_a_surface_raises_in_a_problem_without_manifolds():
    """Only HydrogenTransportProblemDiscontinuous supports manifolds.

    The ``surface`` setter accepts a volume subdomain so a manifold can be passed
    through, so every other problem class has to reject one at initialisation. It would
    otherwise look the subdomain's id up in the *facet* tags and silently report some
    other surface's value -- a plausible number for the wrong surface, not a zero.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    vol = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=1.0, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    left = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[0], 0.0))
    H = F.Species("H", subdomains=[vol])

    model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        species=[H],
        subdomains=[vol, left],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=left, value=1.0, species=H)
        ],
        exports=[F.SurfaceFlux(field=H, surface=vol)],
        temperature=500,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )
    with pytest.raises(TypeError, match="only supported by .*Discontinuous"):
        model.initialise()
