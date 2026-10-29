"""A counter-current membrane contactor: two channels either side of a solid wall.

The geometry of a permeator or a tritium extraction contactor. A 2D solid separator
carries the diffusion, and a 1D fluid channel runs along each face as a codim-1
manifold, exchanging with the solid through a mass transfer coefficient
``J = k (c_fluid - c_wall)``::

        A (feed)          wall          B (sweep)
         x = 0                            x = W
    y=H    ^  c_out       -> J ->    c_in  |
           |                               |
           |          D_wall               |         counter-current:
           |                               |         A flows +y, B flows -y
    y=0    |  c = 1      -> J ->    c_out  v

Each channel is fed at one end with a Dirichlet condition and drains at the other
through a :class:`festim.OutflowBC`. The ends of the wall carry no condition, so they
are sealed, and every atom that enters at A's inlet has to leave through one of the
three other channel ends.

This exercises the whole drift feature set at once: two manifolds with opposed
advection, an outflow at each outlet, and :class:`festim.SurfaceFlux` on the boundary of
a manifold reporting a total flux that includes the drift.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import festim as F

WALL_ID, A_ID, B_ID = 1, 2, 3
A_IN_ID, A_OUT_ID, B_IN_ID, B_OUT_ID = 4, 5, 6, 7

HEIGHT, THICKNESS = 1.0, 0.1
D_WALL, D_FLUID, K_EX, V_FLOW = 0.05, 0.05, 2.0, 1.0
C_FEED = 1.0


def _velocity(mesh, v_y):
    """A uniform vertical velocity, as an ambient 2D vector."""
    velocity = dolfinx.fem.Function(
        dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    )
    velocity.interpolate(
        lambda x: np.vstack([np.zeros(x.shape[1]), np.full(x.shape[1], v_y)])
    )
    return velocity


def _channel(subdomain_id, x_position):
    """A 1D fluid channel running up one face of the wall."""
    return F.VolumeSubdomain(
        id=subdomain_id,
        material=F.Material(D_0=D_FLUID, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], x_position),
    )


def _bottom(x):
    return np.isclose(x[1], 0.0)


def _top(x):
    return np.isclose(x[1], HEIGHT)


def solve_contactor(counter_current: bool, n: int = 80):
    """Channel A always flows ``+y``; channel B flows ``-y`` when counter-current.

    Returns the concentration profiles along both channels and the four end fluxes,
    signed so that positive means leaving the channel.
    """
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([THICKNESS, HEIGHT])],
        [12, n],
    )

    wall = F.VolumeSubdomain(
        id=WALL_ID,
        material=F.Material(D_0=D_WALL, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    channel_a = _channel(A_ID, 0.0)
    channel_b = _channel(B_ID, THICKNESS)

    a_inlet = F.SurfaceSubdomain(id=A_IN_ID, dim=0, locator=_bottom)
    a_outlet = F.SurfaceSubdomain(id=A_OUT_ID, dim=0, locator=_top)
    # B is fed from whichever end it flows away from
    b_inlet = F.SurfaceSubdomain(
        id=B_IN_ID, dim=0, locator=_top if counter_current else _bottom
    )
    b_outlet = F.SurfaceSubdomain(
        id=B_OUT_ID, dim=0, locator=_bottom if counter_current else _top
    )
    v_b = -V_FLOW if counter_current else V_FLOW

    c_wall = F.Species("c_wall", subdomains=[wall])
    c_a = F.Species("c_a", subdomains=[channel_a])
    c_b = F.Species("c_b", subdomains=[channel_b])

    fluxes = {
        "a_inlet": F.SurfaceFlux(field=c_a, surface=a_inlet),
        "a_outlet": F.SurfaceFlux(field=c_a, surface=a_outlet),
        "b_inlet": F.SurfaceFlux(field=c_b, surface=b_inlet),
        "b_outlet": F.SurfaceFlux(field=c_b, surface=b_outlet),
    }

    def exchange_out_of(fluid):
        """What the channel loses to the wall."""
        return F.ParticleSource(
            value=lambda c_f, c_w: -K_EX * (c_f - c_w),
            species=fluid,
            volume=channel_a if fluid is c_a else channel_b,
            species_dependent_value={"c_f": fluid, "c_w": c_wall},
        )

    def exchange_into_wall(fluid, channel):
        """The same quantity, entering the wall. The two must be equal and opposite."""
        return F.ParticleFluxBC(
            subdomain=channel,
            species=c_wall,
            value=lambda c_f, c_w: K_EX * (c_f - c_w),
            species_dependent_value={"c_f": fluid, "c_w": c_wall},
        )

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[c_wall, c_a, c_b],
        subdomains=[wall, channel_a, channel_b, a_inlet, a_outlet, b_inlet, b_outlet],
        sources=[exchange_out_of(c_a), exchange_out_of(c_b)],
        boundary_conditions=[
            exchange_into_wall(c_a, channel_a),
            exchange_into_wall(c_b, channel_b),
            F.FixedConcentrationBC(subdomain=a_inlet, value=C_FEED, species=c_a),
            F.FixedConcentrationBC(subdomain=b_inlet, value=0.0, species=c_b),
            F.OutflowBC(subdomain=a_outlet, species=c_a),
            F.OutflowBC(subdomain=b_outlet, species=c_b),
        ],
        drift_terms=[
            F.AdvectionTerm(
                velocity=_velocity(mesh, V_FLOW), subdomain=channel_a, species=c_a
            ),
            F.AdvectionTerm(
                velocity=_velocity(mesh, v_b), subdomain=channel_b, species=c_b
            ),
        ],
        exports=list(fluxes.values()),
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    model.show_progress_bar = False
    model.initialise()
    model.run()

    def profile(species, manifold):
        function = species.subdomain_to_post_processing_solution[manifold]
        y = function.function_space.tabulate_dof_coordinates()[:, 1]
        order = np.argsort(y)
        return y[order], function.x.array[order]

    _, profile_a = profile(c_a, channel_a)
    _, profile_b = profile(c_b, channel_b)
    return {
        "a": profile_a,  # both sorted by increasing y
        "b": profile_b,
        "flux": {name: export.data[-1] for name, export in fluxes.items()},
        # the concentration the sweep leaves with, ie. what the contactor recovers
        "recovered": profile_b[0] if counter_current else profile_b[-1],
    }


@pytest.fixture(scope="module")
def counter():
    return solve_contactor(counter_current=True)


@pytest.fixture(scope="module")
def co():
    return solve_contactor(counter_current=False)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_every_atom_that_enters_leaves(counter):
    """Nothing accumulates and nothing is lost, to machine precision.

    The wall's own ends carry no condition, so they are sealed, and the exchange with
    the channels is internal. The four channel-end fluxes must therefore sum to zero.
    Because :class:`festim.SurfaceFlux` reports the **total** flux, this is a joint
    check on the drift term, its boundary contribution, the outflow conditions, and the
    export -- get any one of them wrong and the budget does not close.
    """
    flux = counter["flux"]
    assert flux["a_inlet"] < 0, "the feed enters at A's inlet"
    assert flux["a_outlet"] > 0, "the raffinate leaves at A's outlet"
    assert flux["b_outlet"] > 0, "the loaded sweep leaves at B's outlet"

    entering = -flux["a_inlet"]
    assert abs(sum(flux.values())) < 1e-10 * entering, flux


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_each_stream_runs_the_right_way(counter):
    """The feed gives up its load along its flow, and the sweep picks it up along its.

    A flows ``+y`` and so must fall with increasing y. B flows ``-y``, so it must fall
    with *increasing* y as well -- it is at its cleanest where it enters at the top.
    """
    profile_a, profile_b = counter["a"], counter["b"]

    assert np.isclose(profile_a[0], C_FEED, atol=1e-12), "the feed value is imposed"
    assert np.all(np.diff(profile_a) < 0), "the feed is stripped as it rises"

    assert np.isclose(profile_b[-1], 0.0, atol=1e-12), "the sweep enters clean"
    assert np.all(np.diff(profile_b) < 0), "the sweep loads up as it descends"

    # the two streams are counter-current, so the sweep is richest where the feed is
    assert profile_b[0] > profile_b[-1]


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_counter_current_beats_co_current(counter, co):
    """The textbook result, and the reason contactors are plumbed this way.

    Running the sweep against the feed holds a driving force along the whole length,
    rather than letting the two streams approach each other and stall. Both runs are
    identical apart from the direction of B's velocity and which end it is fed from.
    """
    assert counter["recovered"] > co["recovered"]
    # ~3.4% at this duty, and mesh-converged: 3.371 / 3.370 / 3.369 % at n = 60/120/240
    assert counter["recovered"] / co["recovered"] - 1 > 0.02

    # and correspondingly less is left in the feed
    assert counter["a"][-1] < co["a"][-1]

    # the co-current case has to balance too
    assert abs(sum(co["flux"].values())) < 1e-10 * -co["flux"]["a_inlet"]
