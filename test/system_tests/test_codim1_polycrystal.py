"""A manifold adjacent to more than two volume subdomains: the grain-boundary network
of a polycrystal in which every grain is its own subdomain, so the concentration is
allowed to jump from one grain to the next.

Up to two adjacent volumes one integral carries both sides of the manifold, told apart
by the ``"+"``/``"-"`` restrictions. Beyond two that ordering does not exist -- past a
triple junction the network runs between a different pair of grains -- so each adjacent
grain gets an integral of its own on which it is the ``"+"`` side, and a facet shared by
two grains is integrated once per grain. See issue #1208.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F

D_BULK, D_GAMMA = 1.5, 0.7
NETWORK_ID = 100


def strips(n=24, k=(2.0, 5.0, 3.0)):
    """Three vertical strips separated by a two-branch manifold declared as one
    subdomain, so that the manifold is adjacent to all three grains.

    Everything is uniform in y, so the steady state is the 1D series chain::

        c=2 | grain 1 | Gamma_A | grain 2 | Gamma_B | grain 3 | c=0

    Each grain is 1/3 wide, and grain 2 exchanges with *both* branches at the same rate
    -- the exchange law is a property of the (network, grain) pair, not of the branch.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    a, b = 1 / 3, 2 / 3
    grains = [
        F.VolumeSubdomain(
            id=1,
            material=F.Material(D_0=D_BULK, E_D=0.0),
            locator=lambda x: x[0] <= a + 1e-14,
        ),
        F.VolumeSubdomain(
            id=2,
            material=F.Material(D_0=D_BULK, E_D=0.0),
            locator=lambda x: (x[0] >= a - 1e-14) & (x[0] <= b + 1e-14),
        ),
        F.VolumeSubdomain(
            id=3,
            material=F.Material(D_0=D_BULK, E_D=0.0),
            locator=lambda x: x[0] >= b - 1e-14,
        ),
    ]
    network = F.VolumeSubdomain(
        id=NETWORK_ID,
        material=F.Material(D_0=D_GAMMA, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], a) | np.isclose(x[0], b),
    )
    left = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 0.0))
    right = F.SurfaceSubdomain(id=5, locator=lambda x: np.isclose(x[0], 1.0))

    species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])

    # one exchange per grain, each naming only that grain's species -- which is how
    # FESTIM knows which side of the network the term belongs to
    sources = [
        F.ParticleSource(
            value=lambda c_g, c_b, rate=rate: rate * (c_b - c_g),
            species=c_gb,
            volume=network,
            species_dependent_value={"c_b": spe, "c_g": c_gb},
        )
        for spe, rate in zip(species, k, strict=True)
    ]
    bcs = [
        F.ParticleFluxBC(
            subdomain=network,
            species=spe,
            value=lambda c_g, c_b, rate=rate: rate * (c_g - c_b),
            species_dependent_value={"c_b": spe, "c_g": c_gb},
        )
        for spe, rate in zip(species, k, strict=True)
    ]
    bcs += [
        F.FixedConcentrationBC(subdomain=left, value=2.0, species=species[0]),
        F.FixedConcentrationBC(subdomain=right, value=0.0, species=species[2]),
    ]

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[*species, c_gb],
        subdomains=[*grains, network, left, right],
        sources=sources,
        boundary_conditions=bcs,
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    return model, grains, network, species, c_gb


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_three_grains_match_the_series_chain():
    """The steady state of the three-strip chain is closed-form.

    Every exchange and every strip carries the same flux ``J``, so the resistances add::

        J = 2 / (1/D + 1/k1 + 2/k2 + 1/k3)

    the ``2/k2`` because the middle grain exchanges on both of its faces. The rates
    differ from one grain to the next, so a term wired to the wrong grain -- the failure
    mode the ``"+"``/``"-"`` ordering exists to prevent, and which has no ordering to
    rely on beyond two adjacent subdomains -- changes every number below.
    """
    k1, k2, k3 = 2.0, 5.0, 3.0
    model, grains, network, species, c_gb = strips(k=(k1, k2, k3))
    model.initialise()
    model.run()

    j = 2.0 / (1 / D_BULK + 1 / k1 + 2 / k2 + 1 / k3)
    drop = j * (1 / 3) / D_BULK  # across one strip
    c1_right = 2.0 - drop
    c_gamma_a = c1_right - j / k1
    c2_left = c_gamma_a - j / k2
    c2_right = c2_left - drop
    c_gamma_b = c2_right - j / k2
    c3_left = c_gamma_b - j / k3

    c = [s.subdomain_to_post_processing_solution[g] for s, g in zip(species, grains)]
    assert np.isclose(c[0].x.array.max(), 2.0, atol=1e-10)
    assert np.isclose(c[0].x.array.min(), c1_right, atol=1e-8)
    assert np.isclose(c[1].x.array.max(), c2_left, atol=1e-8)
    assert np.isclose(c[1].x.array.min(), c2_right, atol=1e-8)
    assert np.isclose(c[2].x.array.max(), c3_left, atol=1e-8)
    assert np.isclose(c[2].x.array.min(), 0.0, atol=1e-10)

    # the two branches are one function space but two disconnected components, so each
    # sits at its own level
    gb = c_gb.subdomain_to_post_processing_solution[network]
    x = gb.function_space.tabulate_dof_coordinates()[:, 0]
    assert np.allclose(gb.x.array[x < 0.5], c_gamma_a, atol=1e-8)
    assert np.allclose(gb.x.array[x > 0.5], c_gamma_b, atol=1e-8)

    # what enters the first grain leaves the last one: a sign error on one of the six
    # restricted coupling terms breaks this while leaving a plausible profile
    n = ufl.FacetNormal(model.mesh.mesh)
    entity_maps = [sd.cell_map for sd in model.volume_subdomains]

    def wall_flux(spe, grain, surface_id):
        form = dolfinx.fem.form(
            D_BULK
            * ufl.dot(ufl.grad(spe.subdomain_to_solution[grain]), n)
            * model.ds(surface_id),
            entity_maps=entity_maps,
        )
        return model.mesh.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(form), op=MPI.SUM
        )

    assert np.isclose(wall_flux(species[0], grains[0], 4), j, rtol=1e-6)
    assert np.isclose(-wall_flux(species[2], grains[2], 5), j, rtol=1e-6)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_three_grains_allocate_one_measure_id_per_side():
    """Each side gets an integration id of its own, above every id the user declared."""
    model, grains, network, _, _ = strips(n=12)
    model.initialise()

    assert model.manifold_to_volumes[network] == grains
    ids = [model.coupling_measure_id(network, g) for g in grains]
    assert len(set(ids)) == 3
    assert min(ids) > max(sd.id for sd in model.volume_subdomains)
    # each side is the "+" one of its own integral
    assert [model.restriction_of(network, g) for g in grains] == ["+"] * 3


def on_t_junction(x):
    """The T-shaped network: the line y=0.5 plus the branch x=0.5 below it."""
    return np.isclose(x[1], 0.5) | (np.isclose(x[0], 0.5) & (x[1] <= 0.5 + 1e-14))


class TNetwork(F.VolumeSubdomain):
    """The T as one codim-1 subdomain, selected by facet midpoint.

    ``locate_entities`` marks a facet when *all its vertices* satisfy the locator, so
    passing :func:`on_t_junction` directly would also catch the diagonal joining a
    vertex of the vertical branch to a vertex of the horizontal line -- a facet on
    neither, hanging off the junction.
    """

    def locate_subdomain_entities(self, mesh):
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        facet_to_vertex = mesh.topology.connectivity(tdim - 1, 0)
        candidates = dolfinx.mesh.locate_entities(mesh, tdim - 1, on_t_junction)
        x = mesh.geometry.x
        midpoints = np.array(
            [x[facet_to_vertex.links(f)].mean(axis=0) for f in candidates]
        )
        return candidates[on_t_junction(midpoints.T)].astype(np.int32)


def t_junction(n=24, rates=(1.0, 1.0, 1.0)):
    """Three grains meeting at a triple junction, with the whole network -- the line
    y=0.5 and the branch x=0.5 below it -- declared as one connected subdomain.

    Only the bottom-left grain is charged, so the other two are fed through the network.
    ``rates`` is the exchange rate of each grain with it, in declaration order.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    top = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=D_BULK, E_D=0.0),
        locator=lambda x: x[1] >= 0.5 - 1e-14,
    )
    bottom_left = F.VolumeSubdomain(
        id=2,
        material=F.Material(D_0=D_BULK, E_D=0.0),
        locator=lambda x: (x[1] <= 0.5 + 1e-14) & (x[0] <= 0.5 + 1e-14),
    )
    bottom_right = F.VolumeSubdomain(
        id=3,
        material=F.Material(D_0=D_BULK, E_D=0.0),
        locator=lambda x: (x[1] <= 0.5 + 1e-14) & (x[0] >= 0.5 - 1e-14),
    )
    grains = [top, bottom_left, bottom_right]
    network = TNetwork(
        id=NETWORK_ID,
        material=F.Material(D_0=D_GAMMA, E_D=0.0),
        dim=1,
    )
    charged = F.SurfaceSubdomain(
        id=4, locator=lambda x: np.isclose(x[1], 0.0) & (x[0] <= 0.5 + 1e-14)
    )

    species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[*species, c_gb],
        subdomains=[*grains, network, charged],
        sources=[
            F.ParticleSource(
                value=lambda c_g, c_b, rate=rate: rate * (c_b - c_g),
                species=c_gb,
                volume=network,
                species_dependent_value={"c_b": spe, "c_g": c_gb},
            )
            for spe, rate in zip(species, rates, strict=True)
        ],
        boundary_conditions=[
            F.ParticleFluxBC(
                subdomain=network,
                species=spe,
                value=lambda c_g, c_b, rate=rate: rate * (c_g - c_b),
                species_dependent_value={"c_b": spe, "c_g": c_gb},
            )
            for spe, rate in zip(species, rates, strict=True)
        ]
        + [F.FixedConcentrationBC(subdomain=charged, value=2.0, species=species[1])],
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    return model, grains, network, species, c_gb


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_triple_junction_feeds_every_grain_through_one_network():
    """Charging one grain fills the other two, through the junction.

    The grains are separate subdomains with no interface between them, so the network is
    the only path from one to the next; and the two uncharged grains touch it only past
    the junction. Nothing drives a gradient at steady state, so everything must sit at
    the boundary value. A side whose coupling was dropped would leave its grain floating
    instead.
    """
    model, grains, network, species, c_gb = t_junction()
    model.initialise()

    assert model.manifold_to_volumes[network] == grains
    assert model.manifold_is_interior(network)

    # the network is exactly the T and nothing else: 1.0 across plus 0.5 down. A
    # locator passed straight to locate_entities also picks up the diagonal joining
    # the two branches near the junction, which is a facet on neither of them
    sub = network.submesh
    sub.topology.create_connectivity(1, 0)
    cell_to_vertex = sub.topology.connectivity(1, 0)
    length = sum(
        float(np.linalg.norm(np.diff(sub.geometry.x[cell_to_vertex.links(c)], axis=0)))
        for c in range(sub.topology.index_map(1).size_local)
    )
    assert np.isclose(length, 1.5)

    model.run()

    for spe, grain in zip(species, grains, strict=True):
        c = spe.subdomain_to_post_processing_solution[grain].x.array
        assert np.allclose(c, 2.0, atol=1e-8), f"grain {grain.id} did not fill up"
    gb = c_gb.subdomain_to_post_processing_solution[network].x.array
    assert np.allclose(gb, 2.0, atol=1e-8)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_one_grain_can_be_blocked_off_from_the_network():
    """A near-zero exchange rate for a single grain keeps that grain -- and only that
    grain -- empty.

    This is what separate grains buy over a single-subdomain polycrystal, and it is the
    sharpest test of the per-side integrals: the three rates sit on one manifold, and
    the blocked one has to land on the third grain alone. Steady state would hide it,
    since a lone Dirichlet with no sink fills everything eventually whatever the rates,
    so the comparison is made in time.
    """
    model, grains, network, species, c_gb = t_junction(n=16, rates=(1.0, 1.0, 1e-4))
    model.settings.transient = True
    model.settings.final_time = 0.4
    model.settings.stepsize = F.Stepsize(initial_value=0.02)
    model.initialise()
    model.run()

    top, bottom_left, bottom_right = (
        spe.subdomain_to_post_processing_solution[g].x.array
        for spe, g in zip(species, grains, strict=True)
    )
    gb = c_gb.subdomain_to_post_processing_solution[network].x.array

    # the charged grain and the network fill, and the grain coupled normally follows
    assert bottom_left.max() > 1.5
    assert gb.max() > 0.2
    assert top.max() > 0.05
    # the blocked one stays essentially empty, two orders of magnitude below the rest
    assert bottom_right.max() < 1e-2 * top.max()
