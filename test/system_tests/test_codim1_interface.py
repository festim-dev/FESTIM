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
    dS = model.coupling_measure(gamma)

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
