"""Codimensional coupling on an *interior* manifold: a codim-1 subdomain sandwiched
between two bulk subdomains, exchanging flux with both. See issue #1208.

The exchange rates are deliberately different on the two sides throughout. Interior
facets are integrated with a ``dS`` measure whose ``"+"``/``"-"`` ordering is arbitrary
until FESTIM fixes it, and a swapped ordering with equal rates would be invisible.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F

from .tools import error_L2

D_BULK, D_GAMMA = 1.5, 0.7
K_LEFT, K_RIGHT = 2.0, 3.0
LEFT_ID, RIGHT_ID, GAMMA_ID, OUTER_L_ID, OUTER_R_ID = 1, 2, 3, 4, 5


def build(n=20, mesh=None, plane=0.5, swap_declaration_order=False):
    """A left bulk, a right bulk and an interior manifold between them."""
    mesh = (
        mesh
        if mesh is not None
        else dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    )
    left = F.VolumeSubdomain(
        id=LEFT_ID,
        material=F.Material(D_0=D_BULK, E_D=0.0),
        locator=lambda x: x[0] <= plane + 1e-14,
    )
    right = F.VolumeSubdomain(
        id=RIGHT_ID,
        material=F.Material(D_0=D_BULK, E_D=0.0),
        locator=lambda x: x[0] >= plane - 1e-14,
    )
    gamma = F.VolumeSubdomain(
        id=GAMMA_ID,
        material=F.Material(D_0=D_GAMMA, E_D=0.0),
        dim=mesh.topology.dim - 1,
        locator=lambda x: np.isclose(x[0], plane),
    )
    outer_l = F.SurfaceSubdomain(id=OUTER_L_ID, locator=lambda x: np.isclose(x[0], 0.0))
    outer_r = F.SurfaceSubdomain(id=OUTER_R_ID, locator=lambda x: np.isclose(x[0], 1.0))

    H_l = F.Species("H_l", subdomains=[left])
    H_r = F.Species("H_r", subdomains=[right])
    H_g = F.Species("H_g", subdomains=[gamma])

    # one exchange per side, each naming only that side's bulk species -- which is how
    # FESTIM knows which side of the interface the term belongs to
    sources = [
        F.ParticleSource(
            value=lambda c_g, c_b: K_LEFT * (c_b - c_g),
            species=H_g,
            volume=gamma,
            species_dependent_value={"c_b": H_l, "c_g": H_g},
        ),
        F.ParticleSource(
            value=lambda c_g, c_b: K_RIGHT * (c_b - c_g),
            species=H_g,
            volume=gamma,
            species_dependent_value={"c_b": H_r, "c_g": H_g},
        ),
    ]
    bcs = [
        F.ParticleFluxBC(
            subdomain=gamma,
            species=H_l,
            value=lambda c_g, c_b: K_LEFT * (c_g - c_b),
            species_dependent_value={"c_b": H_l, "c_g": H_g},
        ),
        F.ParticleFluxBC(
            subdomain=gamma,
            species=H_r,
            value=lambda c_g, c_b: K_RIGHT * (c_g - c_b),
            species_dependent_value={"c_b": H_r, "c_g": H_g},
        ),
        F.FixedConcentrationBC(subdomain=outer_l, value=2.0, species=H_l),
        F.FixedConcentrationBC(subdomain=outer_r, value=0.0, species=H_r),
    ]

    subdomains = [right, left] if swap_declaration_order else [left, right]
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_l, H_r, H_g],
        subdomains=[*subdomains, gamma, outer_l, outer_r],
        sources=sources,
        boundary_conditions=bcs,
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    return model, (left, right, gamma), (H_l, H_r, H_g)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_interior_manifold_matches_analytical_solution():
    """A 1D steady problem across the interface has a closed-form solution.

    With c=2 at x=0, c=0 at x=1 and a manifold at x=0.5, every flux equals the same J
    at steady state::

        D/0.5 (2 - c_L) = K_LEFT (c_L - c_G) = K_RIGHT (c_G - c_R) = D/0.5 c_R = J

    which gives J = 4/3, c_L = 14/9, c_G = 8/9, c_R = 4/9. Swapping the two sides would
    give c_G = 10/9 instead, so this pins the "+"/"-" ordering as well as the physics.
    """
    model, (left, right, gamma), (H_l, H_r, H_g) = build()
    model.initialise()
    model.run()

    c_l = H_l.subdomain_to_post_processing_solution[left].x.array
    c_r = H_r.subdomain_to_post_processing_solution[right].x.array
    c_g = H_g.subdomain_to_post_processing_solution[gamma].x.array

    assert np.isclose(c_l.max(), 2.0, atol=1e-10)
    assert np.isclose(c_l.min(), 14 / 9, atol=1e-8)
    assert np.isclose(c_r.max(), 4 / 9, atol=1e-8)
    assert np.isclose(c_r.min(), 0.0, atol=1e-10)
    assert np.allclose(c_g, 8 / 9, atol=1e-8)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_interior_manifold_ordering_is_independent_of_declaration_order():
    """Which bulk subdomain is "+" must not depend on the order the user declares them.

    DOLFINx's own ordering of the two cells of an interior facet is arbitrary, so this
    is the test that catches the restrictions being wired up to the wrong sides.
    """
    results = []
    for swap in (False, True):
        model, (left, _, gamma), (H_l, _, H_g) = build(swap_declaration_order=swap)
        model.initialise()
        model.run()
        results.append(
            (
                H_g.subdomain_to_post_processing_solution[gamma].x.array.copy(),
                H_l.subdomain_to_post_processing_solution[left].x.array.copy(),
            )
        )

    assert np.allclose(results[0][0], results[1][0], atol=1e-12)
    assert np.allclose(results[0][1], results[1][1], atol=1e-12)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_interior_manifold_conserves_particles():
    """What leaves the left bulk enters the manifold and leaves through the right.

    A sign error in one of the two restricted coupling terms breaks conservation while
    leaving a plausible-looking solution, so this catches what a convergence study can
    miss.
    """
    model, (left, right, gamma), (H_l, H_r, H_g) = build()
    model.initialise()
    model.run()

    c_l = H_l.subdomain_to_solution[left]
    c_r = H_r.subdomain_to_solution[right]
    c_g = H_g.subdomain_to_solution[gamma]
    entity_maps = [sd.cell_map for sd in model.volume_subdomains]
    dS = model.facet_measure(gamma)(gamma.id)

    def assemble(form):
        return model.mesh.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(form, entity_maps=entity_maps)
            ),
            op=MPI.SUM,
        )

    r_left = model.restriction_of(gamma, left)
    r_right = model.restriction_of(gamma, right)
    into_gamma_from_left = assemble((K_LEFT * (c_l - c_g))(r_left) * dS)
    out_of_gamma_to_right = assemble((K_RIGHT * (c_g - c_r))(r_right) * dS)

    # steady state: everything the manifold takes in on one side it gives up on the
    # other, and both equal the flux crossing the domain. FESTIM's flux convention is
    # the influx, D grad(c).n with n the outward normal
    n = ufl.FacetNormal(model.mesh.mesh)
    through_left_wall = assemble(
        D_BULK * ufl.dot(ufl.grad(c_l), n) * model.ds(OUTER_L_ID)
    )
    assert np.isclose(into_gamma_from_left, out_of_gamma_to_right, rtol=1e-8)
    assert np.isclose(into_gamma_from_left, through_left_wall, rtol=1e-6)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_interior_manifold_inside_a_single_subdomain():
    """A manifold can sit inside one volume subdomain rather than between two.

    That is the grain-boundary network of a single-phase polycrystal: every grain is the
    same material, so one bulk subdomain lies on *both* sides of the manifold. Whether
    the coupling is an interior-facet integral is a property of the mesh, not of how
    many subdomains the manifold separates -- deciding it from the subdomain count picks
    ``ds``, which integrates to exactly zero over interior facets and leaves the
    coupling silently doing nothing.
    """
    n, plane = 16, 0.5
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    bulk = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=D_BULK, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    gamma = F.VolumeSubdomain(
        id=GAMMA_ID,
        material=F.Material(D_0=D_GAMMA, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], plane),
    )
    outer = F.SurfaceSubdomain(id=OUTER_L_ID, locator=lambda x: np.isclose(x[0], 0.0))

    H_b = F.Species("H_b", subdomains=[bulk])
    H_g = F.Species("H_g", subdomains=[gamma])

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_b, H_g],
        subdomains=[bulk, gamma, outer],
        sources=[
            F.ParticleSource(
                value=lambda c_g, c_b: K_LEFT * (c_b - c_g),
                species=H_g,
                volume=gamma,
                species_dependent_value={"c_b": H_b, "c_g": H_g},
            )
        ],
        boundary_conditions=[
            F.ParticleFluxBC(
                subdomain=gamma,
                species=H_b,
                value=lambda c_g, c_b: K_LEFT * (c_g - c_b),
                species_dependent_value={"c_b": H_b, "c_g": H_g},
            ),
            F.FixedConcentrationBC(subdomain=outer, value=2.0, species=H_b),
        ],
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    model.initialise()

    assert model.manifold_is_interior(gamma)
    assert model.facet_measure(gamma)(gamma.id).integral_type() == "interior_facet"

    model.run()

    # nothing drives a gradient, so the bulk sits at its boundary value and the manifold
    # is pulled all the way to it. With the coupling silently dropped the manifold would
    # keep its initial value of zero instead.
    c_b = H_b.subdomain_to_post_processing_solution[bulk].x.array
    c_g = H_g.subdomain_to_post_processing_solution[gamma].x.array
    assert np.allclose(c_b, 2.0, atol=1e-10)
    assert np.allclose(c_g, 2.0, atol=1e-8)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_manifold_mixing_interior_and_exterior_facets_raises():
    """A manifold has to be wholly inside the mesh or wholly on its boundary: the two
    need different measures, and one of them would be silently dropped."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    bulk = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=D_BULK, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    # the line x=0.5 is interior, the line x=0 is on the boundary
    gamma = F.VolumeSubdomain(
        id=GAMMA_ID,
        material=F.Material(D_0=D_GAMMA, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], 0.5) | np.isclose(x[0], 0.0),
    )
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[
            F.Species("H_b", subdomains=[bulk]),
            F.Species("H_g", subdomains=[gamma]),
        ],
        subdomains=[bulk, gamma],
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    with pytest.raises(ValueError, match=r"interior and \d+ exterior facets"):
        model.initialise()


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_interior_manifold_transient_reaches_steady_state():
    """The same interface, run in time from a zero initial condition.

    A manifold's time derivative is integrated over its submesh and so needs a
    submesh-resident ``dt``; with the parent-mesh one the form does not compile at all
    (see ``test_transient_manifold_integrates_dt_exactly``). Left long enough the
    transient must land on the closed-form steady state of
    ``test_interior_manifold_matches_analytical_solution``.
    """
    model, (left, right, gamma), (H_l, H_r, H_g) = build()
    model.settings.transient = True
    model.settings.final_time = 8.0
    model.settings.stepsize = F.Stepsize(initial_value=0.05)
    model.initialise()
    model.run()

    c_l = H_l.subdomain_to_post_processing_solution[left].x.array
    c_r = H_r.subdomain_to_post_processing_solution[right].x.array
    c_g = H_g.subdomain_to_post_processing_solution[gamma].x.array

    assert np.isclose(c_l.min(), 14 / 9, atol=1e-5)
    assert np.isclose(c_r.max(), 4 / 9, atol=1e-5)
    assert np.allclose(c_g, 8 / 9, atol=1e-5)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_source_spanning_both_sides_raises():
    """A single manifold source cannot read both sides: it has no single restriction."""
    model, (_, _, gamma), (H_l, H_r, H_g) = build()
    model.sources.append(
        F.ParticleSource(
            value=lambda c_l, c_r: c_l + c_r,
            species=H_g,
            volume=gamma,
            species_dependent_value={"c_l": H_l, "c_r": H_r},
        )
    )
    with pytest.raises(ValueError, match="both of its sides"):
        model.initialise()


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_interface_and_manifold_on_the_same_facets_raises():
    """A concentration jump or a manifold equation, across a pair of volumes -- not
    both."""
    model, (left, right, _), _ = build()
    model.interfaces = [F.Interface(id=9, subdomains=[left, right])]
    with pytest.raises(ValueError, match=r"share \d+ facets"):
        model.initialise()


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_interior_manifold_3d_tilted_mms():
    """MMS on a tilted 2-manifold sandwiched inside a 3D mesh.

    Manufactured in the material coordinates ``xi = R^T X`` so that a rotation, being an
    isometry, carries the solution over exactly::

        c_left(xi)  = a_l + b_l xi_0 + cos(pi xi_1)
        c_right(xi) = a_r + b_r xi_0 + cos(pi xi_1)
        c_gamma(xi) = a_g + cos(pi xi_1)

    The slopes are fixed by the two flux conditions at the manifold, which is what makes
    the exchange terms constant and the manufactured source closed-form::

        b_l (D + k_l p) = k_l (a_g - a_l)      b_r (k_r p - D) = k_r (a_g - a_r)

    All three fields share the same ``cos(pi xi_1)`` profile, so ``d/d xi_1`` vanishes
    at ``xi_1 = 0, 1`` and no condition is needed on the remaining faces or on the
    edges of the manifold -- strong BCs on a manifold are not supported.

    The exchange rates differ between the two sides, so a swapped ``"+"``/``"-"``
    ordering would not reproduce this solution.
    """
    pi = np.pi
    plane = 0.5
    k_l, k_r = 2.0, 4.0
    a_l, a_g, a_r = 2.0, 1.0, 0.5
    b_l = k_l * (a_g - a_l) / (D_BULK + k_l * plane)
    b_r = k_r * (a_g - a_r) / (k_r * plane - D_BULK)
    # constant exchange fluxes into the manifold from either side
    j_l = k_l * (a_l + b_l * plane - a_g)
    j_r = k_r * (a_r + b_r * plane - a_g)

    def rotation(alpha=0.6, gamma_=0.4):
        ca, sa, cg, sg = np.cos(alpha), np.sin(alpha), np.cos(gamma_), np.sin(gamma_)
        Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cg, 0.0, sg], [0.0, 1.0, 0.0], [-sg, 0.0, cg]])
        return Ry @ Rz

    def run(n):
        R = rotation()
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
        mesh.geometry.x[:, :] = mesh.geometry.x @ R.T
        Rt_np = R.T
        Rt = ufl.as_matrix(Rt_np.tolist())

        def xi(x):
            return Rt * ufl.as_vector([x[0], x[1], x[2]])

        def material(x):
            return Rt_np @ x[:3]

        left = F.VolumeSubdomain(
            id=LEFT_ID,
            material=F.Material(D_0=D_BULK, E_D=0.0),
            locator=lambda x: material(x)[0] <= plane + 1e-12,
        )
        right = F.VolumeSubdomain(
            id=RIGHT_ID,
            material=F.Material(D_0=D_BULK, E_D=0.0),
            locator=lambda x: material(x)[0] >= plane - 1e-12,
        )
        gamma = F.VolumeSubdomain(
            id=GAMMA_ID,
            material=F.Material(D_0=D_GAMMA, E_D=0.0),
            dim=2,
            locator=lambda x: np.isclose(material(x)[0], plane),
        )
        outer_l = F.SurfaceSubdomain(
            id=OUTER_L_ID, locator=lambda x: np.isclose(material(x)[0], 0.0)
        )
        outer_r = F.SurfaceSubdomain(
            id=OUTER_R_ID, locator=lambda x: np.isclose(material(x)[0], 1.0)
        )

        H_l = F.Species("H_l", subdomains=[left])
        H_r = F.Species("H_r", subdomains=[right])
        H_g = F.Species("H_g", subdomains=[gamma])

        sources = [
            F.ParticleSource(
                value=lambda x: D_BULK * pi**2 * ufl.cos(pi * xi(x)[1]),
                species=H_l,
                volume=left,
            ),
            F.ParticleSource(
                value=lambda x: D_BULK * pi**2 * ufl.cos(pi * xi(x)[1]),
                species=H_r,
                volume=right,
            ),
            F.ParticleSource(
                value=lambda x: D_GAMMA * pi**2 * ufl.cos(pi * xi(x)[1]) - j_l - j_r,
                species=H_g,
                volume=gamma,
            ),
            F.ParticleSource(
                value=lambda c_g, c_b: k_l * (c_b - c_g),
                species=H_g,
                volume=gamma,
                species_dependent_value={"c_b": H_l, "c_g": H_g},
            ),
            F.ParticleSource(
                value=lambda c_g, c_b: k_r * (c_b - c_g),
                species=H_g,
                volume=gamma,
                species_dependent_value={"c_b": H_r, "c_g": H_g},
            ),
        ]
        bcs = [
            F.ParticleFluxBC(
                subdomain=gamma,
                species=H_l,
                value=lambda c_g, c_b: k_l * (c_g - c_b),
                species_dependent_value={"c_b": H_l, "c_g": H_g},
            ),
            F.ParticleFluxBC(
                subdomain=gamma,
                species=H_r,
                value=lambda c_g, c_b: k_r * (c_g - c_b),
                species_dependent_value={"c_b": H_r, "c_g": H_g},
            ),
            F.FixedConcentrationBC(
                subdomain=outer_l,
                species=H_l,
                value=lambda x: a_l + b_l * xi(x)[0] + ufl.cos(pi * xi(x)[1]),
            ),
            F.FixedConcentrationBC(
                subdomain=outer_r,
                species=H_r,
                value=lambda x: a_r + b_r * xi(x)[0] + ufl.cos(pi * xi(x)[1]),
            ),
        ]

        model = F.HydrogenTransportProblemDiscontinuous(
            mesh=F.Mesh(mesh),
            species=[H_l, H_r, H_g],
            subdomains=[left, right, gamma, outer_l, outer_r],
            sources=sources,
            boundary_conditions=bcs,
            temperature=500,
            settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
        )
        model.initialise()
        model.run()

        def exact(sub, a, b):
            X = Rt * ufl.SpatialCoordinate(sub)
            return a + b * X[0] + ufl.cos(pi * X[1])

        return (
            error_L2(
                H_l.subdomain_to_post_processing_solution[left],
                exact(left.submesh, a_l, b_l),
            ),
            error_L2(
                H_r.subdomain_to_post_processing_solution[right],
                exact(right.submesh, a_r, b_r),
            ),
            error_L2(
                H_g.subdomain_to_post_processing_solution[gamma],
                exact(gamma.submesh, a_g, 0.0),
            ),
        )

    refinements = [4, 8, 16]
    errors = [run(n) for n in refinements]
    for field in range(3):
        rates = [
            np.log(errors[i][field] / errors[i + 1][field])
            / np.log(refinements[i + 1] / refinements[i])
            for i in range(len(refinements) - 1)
        ]
        assert all(r > 1.8 for r in rates), (field, rates)


# --- several interior-facet integrands in one form -------------------------------
#
# An interior manifold's coupling and an Interface's continuity condition are both dS
# integrals of the parent mesh, and both need integration data of their own. They
# cannot each carry their own measure: UFL collects one subdomain_data entry per
# integral of a form, and dolfinx.fem.form asserts they are all the same object before
# using the first for every id. Any volume subdomain bounded by two of them at once
# therefore exercises the shared measure.

K1_LEFT, K1_RIGHT, K2_LEFT, K2_RIGHT = 2.0, 3.0, 4.0, 5.0
MID_ID, GAMMA_1_ID, GAMMA_2_ID, INTERFACE_ID = 6, 7, 8, 9


def _exchange(gamma, gamma_species, bulk_species, k):
    # NOTE perhaps having a small helper like this could be nice to have in the
    # base code at some point
    """The two halves of a codimensional coupling: a source feeding the manifold and
    the matching flux leaving the bulk."""
    dependencies = {"c_b": bulk_species, "c_g": gamma_species}
    return (
        F.ParticleSource(
            value=lambda c_g, c_b: k * (c_b - c_g),
            species=gamma_species,
            volume=gamma,
            species_dependent_value=dependencies,
        ),
        F.ParticleFluxBC(
            subdomain=gamma,
            species=bulk_species,
            value=lambda c_g, c_b: k * (c_g - c_b),
            species_dependent_value=dependencies,
        ),
    )


def build_two_junctions(n=12, second_junction="manifold"):
    """Three vertical strips, joined at ``x = 1/3`` by an interior manifold and at
    ``x = 2/3`` by either a second manifold or an :class:`festim.Interface`.

    Either way the middle strip is bounded by two interior-facet integrands at once, so
    both of their ``dS`` integrals land in its form.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    mat = F.Material(D_0=D_BULK, E_D=0.0, K_S_0=1.0, E_K_S=0.0)
    mat_gamma = F.Material(D_0=D_GAMMA, E_D=0.0)

    left = F.VolumeSubdomain(
        id=LEFT_ID, material=mat, locator=lambda x: x[0] <= 1 / 3 + 1e-14
    )
    middle = F.VolumeSubdomain(
        id=MID_ID,
        material=mat,
        locator=lambda x: np.logical_and(x[0] >= 1 / 3 - 1e-14, x[0] <= 2 / 3 + 1e-14),
    )
    right = F.VolumeSubdomain(
        id=RIGHT_ID, material=mat, locator=lambda x: x[0] >= 2 / 3 - 1e-14
    )
    gamma_1 = F.VolumeSubdomain(
        id=GAMMA_1_ID,
        material=mat_gamma,
        dim=1,
        locator=lambda x: np.isclose(x[0], 1 / 3),
    )
    outer_l = F.SurfaceSubdomain(id=OUTER_L_ID, locator=lambda x: np.isclose(x[0], 0.0))
    outer_r = F.SurfaceSubdomain(id=OUTER_R_ID, locator=lambda x: np.isclose(x[0], 1.0))

    H_l = F.Species("H_l", subdomains=[left])
    H_g1 = F.Species("H_g1", subdomains=[gamma_1])
    interfaces, sources, bcs = [], [], []

    if second_junction == "manifold":
        gamma_2 = F.VolumeSubdomain(
            id=GAMMA_2_ID,
            material=mat_gamma,
            dim=1,
            locator=lambda x: np.isclose(x[0], 2 / 3),
        )
        H_m = F.Species("H_m", subdomains=[middle])
        H_r = F.Species("H_r", subdomains=[right])
        H_g2 = F.Species("H_g2", subdomains=[gamma_2])
        extra = [gamma_2]
        species = [H_l, H_m, H_r, H_g1, H_g2]
        for bulk_species, k in ((H_m, K2_LEFT), (H_r, K2_RIGHT)):
            source, flux = _exchange(gamma_2, H_g2, bulk_species, k)
            sources.append(source)
            bcs.append(flux)
        outflow_species = H_r
    else:
        # an Interface needs its species defined on both of its subdomains, so the
        # middle and right strips share one
        H_m = F.Species("H_m", subdomains=[middle, right])
        extra = []
        species = [H_l, H_m, H_g1]
        interfaces = [F.Interface(id=INTERFACE_ID, subdomains=[middle, right])]
        outflow_species = H_m

    for bulk_species, k in ((H_l, K1_LEFT), (H_m, K1_RIGHT)):
        source, flux = _exchange(gamma_1, H_g1, bulk_species, k)
        sources.append(source)
        bcs.append(flux)

    bcs += [
        F.FixedConcentrationBC(subdomain=outer_l, value=2.0, species=H_l),
        F.FixedConcentrationBC(subdomain=outer_r, value=0.0, species=outflow_species),
    ]

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=species,
        subdomains=[left, middle, right, gamma_1, *extra, outer_l, outer_r],
        sources=sources,
        boundary_conditions=bcs,
        interfaces=interfaces,
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    return model, (left, middle, right, gamma_1, *extra), species


def assert_subdomain_data_is_shared(form, name):
    """Check DOLFINx's own invariant, with a message that says what broke.

    ``dolfinx.fem.form`` asserts that all the integrals of a form sharing a domain and
    an integral type carry the *same* ``subdomain_data`` object, then uses the first of
    them for every subdomain id. A violation surfaces either as a bare
    ``AssertionError`` deep inside the compiler or, with assertions disabled, as one
    manifold silently integrated over another's facets.
    """
    if not isinstance(form, ufl.Form):
        # a manifold carrying no coupling term contributes nothing to the parent mesh
        return
    for per_type in form.subdomain_data().values():
        for integral_type, data in per_type.items():
            distinct = {id(d) for d in data if d is not None}
            assert len(distinct) <= 1, (
                f"the {integral_type} integrals of {name} carry {len(distinct)} "
                "different subdomain_data objects; they must share one"
            )


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_two_interior_manifolds_bounding_one_subdomain():
    """A strip between two grain boundaries couples to both at once.

    At steady state the chain is a series of resistances carrying one flux J::

        2 - 0 = J (1/(3D) + 1/k1L + 1/k1R + 1/(3D) + 1/k2L + 1/k2R + 1/(3D))

    Each manifold is uniform along its length, so its own diffusion term drops out and
    the concentrations are pinned exactly. All four exchange rates differ, so a swapped
    restriction -- or one manifold integrated over the other's facets -- moves them.
    """
    model, subdomains, species = build_two_junctions(second_junction="manifold")
    left, middle, right, gamma_1, gamma_2 = subdomains
    H_l, H_m, H_r, H_g1, H_g2 = species
    model.initialise()

    for subdomain in subdomains:
        assert_subdomain_data_is_shared(subdomain.F, f"volume subdomain {subdomain.id}")

    # sharing means the same object, not merely equal data
    shared = model.interior_facet_measure
    assert model.facet_measure(gamma_1) is shared
    assert model.facet_measure(gamma_2) is shared
    assert {tag for tag, _ in shared.subdomain_data()} == {GAMMA_1_ID, GAMMA_2_ID}

    model.run()

    # walk the resistance chain back from the empty wall on the right
    bulk_r = 1 / (3 * D_BULK)
    flux = 2.0 / (3 * bulk_r + 1 / K1_LEFT + 1 / K1_RIGHT + 1 / K2_LEFT + 1 / K2_RIGHT)
    c_right_max = flux * bulk_r
    c_gamma_2 = c_right_max + flux / K2_RIGHT
    c_mid_min = c_gamma_2 + flux / K2_LEFT
    c_mid_max = c_mid_min + flux * bulk_r
    c_gamma_1 = c_mid_max + flux / K1_RIGHT
    c_left_min = c_gamma_1 + flux / K1_LEFT

    def values(spe, subdomain):
        return spe.subdomain_to_post_processing_solution[subdomain].x.array

    assert np.isclose(values(H_l, left).max(), 2.0, atol=1e-10)
    assert np.isclose(values(H_l, left).min(), c_left_min, atol=1e-8)
    assert np.allclose(values(H_g1, gamma_1), c_gamma_1, atol=1e-8)
    assert np.isclose(values(H_m, middle).max(), c_mid_max, atol=1e-8)
    assert np.isclose(values(H_m, middle).min(), c_mid_min, atol=1e-8)
    assert np.allclose(values(H_g2, gamma_2), c_gamma_2, atol=1e-8)
    assert np.isclose(values(H_r, right).max(), c_right_max, atol=1e-8)
    assert np.isclose(values(H_r, right).min(), 0.0, atol=1e-10)


# --- a bulk species that lives on more than one subdomain -------------------------


def build_shared_bulk_species(n=12):
    """Three strips with a manifold between the first two, where the bulk species of
    the second strip also lives on the third.

    A species spanning several subdomains has a solution on each, so the coupling
    source on the manifold has to be told which one it reads -- the side of the
    manifold the term belongs to. Nothing couples the middle and right strips here, so
    at steady state no flux crosses the manifold and everything upstream of it settles
    at the fed wall's value.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    mat = F.Material(D_0=D_BULK, E_D=0.0)

    left = F.VolumeSubdomain(
        id=LEFT_ID, material=mat, locator=lambda x: x[0] <= 1 / 3 + 1e-14
    )
    middle = F.VolumeSubdomain(
        id=MID_ID,
        material=mat,
        locator=lambda x: np.logical_and(x[0] >= 1 / 3 - 1e-14, x[0] <= 2 / 3 + 1e-14),
    )
    right = F.VolumeSubdomain(
        id=RIGHT_ID, material=mat, locator=lambda x: x[0] >= 2 / 3 - 1e-14
    )
    gamma = F.VolumeSubdomain(
        id=GAMMA_1_ID,
        material=F.Material(D_0=D_GAMMA, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], 1 / 3),
    )
    outer_l = F.SurfaceSubdomain(id=OUTER_L_ID, locator=lambda x: np.isclose(x[0], 0.0))
    outer_r = F.SurfaceSubdomain(id=OUTER_R_ID, locator=lambda x: np.isclose(x[0], 1.0))

    H_l = F.Species("H_l", subdomains=[left])
    H_mr = F.Species("H_mr", subdomains=[middle, right])
    H_g = F.Species("H_g", subdomains=[gamma])

    sources, bcs = [], []
    for bulk_species, k in ((H_l, K1_LEFT), (H_mr, K1_RIGHT)):
        source, flux = _exchange(gamma, H_g, bulk_species, k)
        sources.append(source)
        bcs.append(flux)
    bcs += [
        F.FixedConcentrationBC(subdomain=outer_l, value=2.0, species=H_l),
        F.FixedConcentrationBC(subdomain=outer_r, value=0.0, species=H_mr),
    ]

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_l, H_mr, H_g],
        subdomains=[left, middle, right, gamma, outer_l, outer_r],
        sources=sources,
        boundary_conditions=bcs,
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    return model, (left, middle, right, gamma), (H_l, H_mr, H_g)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_coupling_reads_the_species_solution_on_the_manifold_s_own_side():
    """``H_mr`` has a solution on the middle strip and another on the right one, and
    the exchange with the manifold reads the middle.

    Reading the right one instead would leave the two halves of the exchange
    disagreeing -- the source on the manifold pulling towards 0 while the flux out of
    the middle strip pushes towards its own concentration -- and nothing would
    equilibrate at the fed wall's value.
    """
    model, (left, middle, right, gamma), (H_l, H_mr, H_g) = build_shared_bulk_species()
    model.initialise()
    model.run()

    def values(spe, subdomain):
        return spe.subdomain_to_post_processing_solution[subdomain].x.array

    # no outlet past the manifold, so no flux crosses it and everything upstream sits
    # at the fed wall's value
    assert np.allclose(values(H_l, left), 2.0, atol=1e-8)
    assert np.allclose(values(H_g, gamma), 2.0, atol=1e-8)
    assert np.allclose(values(H_mr, middle), 2.0, atol=1e-8)
    # the right strip shares the species but nothing else, and only sees its own wall
    assert np.allclose(values(H_mr, right), 0.0, atol=1e-8)


# --- an interface and an interior manifold in one model ---------------------------
#
# The two are independent mechanisms on different facets, but they meet in the form of
# any volume subdomain bounded by both: an interface condition and a manifold coupling
# are both dS integrals of the parent mesh, so that form needs the shared measure.

PENALTY = 10.0


def build_interface_and_manifold(n=12):
    """Three strips joined at ``x = 1/3`` by an interior manifold and at ``x = 2/3`` by
    an :class:`festim.Interface`, so the middle strip is bounded by both."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    # the interface relates u_0/K_0 to u_1/K_1, so the bulks need a solubility
    mat = F.Material(D_0=D_BULK, E_D=0.0, K_S_0=1.0, E_K_S=0.0)

    left = F.VolumeSubdomain(
        id=LEFT_ID, material=mat, locator=lambda x: x[0] <= 1 / 3 + 1e-14
    )
    middle = F.VolumeSubdomain(
        id=MID_ID,
        material=mat,
        locator=lambda x: np.logical_and(x[0] >= 1 / 3 - 1e-14, x[0] <= 2 / 3 + 1e-14),
    )
    right = F.VolumeSubdomain(
        id=RIGHT_ID, material=mat, locator=lambda x: x[0] >= 2 / 3 - 1e-14
    )
    gamma = F.VolumeSubdomain(
        id=GAMMA_1_ID,
        material=F.Material(D_0=D_GAMMA, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], 1 / 3),
    )
    outer_l = F.SurfaceSubdomain(id=OUTER_L_ID, locator=lambda x: np.isclose(x[0], 0.0))
    outer_r = F.SurfaceSubdomain(id=OUTER_R_ID, locator=lambda x: np.isclose(x[0], 1.0))

    H_l = F.Species("H_l", subdomains=[left])
    # the interface enforces the continuity of this one across x = 2/3
    H_mr = F.Species("H_mr", subdomains=[middle, right])
    H_g = F.Species("H_g", subdomains=[gamma])

    sources, bcs = [], []
    for bulk_species, k in ((H_l, K1_LEFT), (H_mr, K1_RIGHT)):
        source, flux = _exchange(gamma, H_g, bulk_species, k)
        sources.append(source)
        bcs.append(flux)
    bcs += [
        F.FixedConcentrationBC(subdomain=outer_l, value=2.0, species=H_l),
        F.FixedConcentrationBC(subdomain=outer_r, value=0.0, species=H_mr),
    ]

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_l, H_mr, H_g],
        subdomains=[left, middle, right, gamma, outer_l, outer_r],
        sources=sources,
        boundary_conditions=bcs,
        interfaces=[
            F.Interface(id=9, subdomains=[middle, right], penalty_term=PENALTY)
        ],
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    return model, (left, middle, right, gamma), (H_l, H_mr, H_g)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_interface_and_interior_manifold_bounding_one_subdomain(recwarn):
    """A strip may have a manifold on one side and an interface on the other.

    What the two may not do is cover the *same* facets, which is checked separately.

    One flux crosses everything at steady state, so the drop across each element of the
    chain predicts the same J. That holds whatever the interface's own resistance is,
    which is what makes it a fair check of the coupling rather than of the penalty
    constant.
    """
    model, (left, middle, right, gamma), (H_l, H_mr, H_g) = (
        build_interface_and_manifold()
    )
    model.initialise()

    # neither the left bulk species nor the manifold's own species touches both sides
    # species the interface could not couple. Matched on the phrase unique to that
    # warning: initialise() emits unrelated ones that also mention interfaces.
    skipped = [
        str(w.message)
        for w in recwarn
        if "the interface condition is not applied" in str(w.message)
    ]
    assert skipped == []

    # the middle strip's form now carries a manifold dS and an interface dS
    assert_subdomain_data_is_shared(middle.F, "the middle strip")

    model.run()

    def values(spe, subdomain):
        return spe.subdomain_to_post_processing_solution[subdomain].x.array

    c_left, c_g = values(H_l, left), values(H_g, gamma)
    c_middle, c_right = values(H_mr, middle), values(H_mr, right)

    # monotone step down from the fed wall to the empty one
    assert np.isclose(c_left.max(), 2.0, atol=1e-10)
    assert c_left.min() > c_g.max()
    assert c_g.min() > c_middle.max()
    assert c_middle.min() > c_right.max()
    assert np.isclose(c_right.min(), 0.0, atol=1e-10)

    # the same flux through both bulk strips and both faces of the manifold
    bulk_conductance = 3 * D_BULK
    fluxes = [
        bulk_conductance * (2.0 - c_left.min()),
        K1_LEFT * (c_left.min() - c_g.max()),
        K1_RIGHT * (c_g.min() - c_middle.max()),
        bulk_conductance * (c_middle.max() - c_middle.min()),
        bulk_conductance * (c_right.max() - 0.0),
    ]
    assert np.allclose(fluxes, fluxes[0], rtol=1e-6)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_species_on_one_side_of_an_interface_warns():
    """A species on exactly one of an interface's subdomains cannot be made continuous
    across it. That is either deliberate -- a species absent from the neighbouring
    material -- or a subdomain missing from its list, and only the user can tell which,
    so it is skipped out loud."""
    model, (_, _, right, _), _ = build_interface_and_manifold()
    outer_r = next(s for s in model.subdomains if s.id == OUTER_R_ID)
    confined = F.Species("H_right_only", subdomains=[right])
    model.species.append(confined)
    model.boundary_conditions.append(
        F.FixedConcentrationBC(subdomain=outer_r, value=1.0, species=confined)
    )

    with pytest.warns(UserWarning, match="the other side of interface 9"):
        model.initialise()
