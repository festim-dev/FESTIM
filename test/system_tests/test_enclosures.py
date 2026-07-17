from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F

from ..markers import requires_dolfinx_011

pytestmark = requires_dolfinx_011


def make_model(n=16, enclosures=None, final_time=20.0, dt=1.0):
    """A 1D slab with a single material and no boundary conditions, to which enclosures
    can be attached. The species field is decoupled unless a coupling BC is given."""
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, n)
    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh=mesh)

    material = F.Material(name="mat", D_0=1e-9, E_D=0.0)
    volume = F.VolumeSubdomain1D(id=1, borders=[0.0, 1.0], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=1.0)
    my_model.subdomains = [volume, left, right]

    H = F.Species("H", subdomains=[volume])
    my_model.species = [H]
    my_model.temperature = 500.0
    my_model.enclosures = enclosures or []
    my_model.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=True,
        final_time=final_time,
        stepsize=F.Stepsize(dt),
    )
    my_model.show_progress_bar = False
    return my_model, volume, left, right, H


def backward_euler_decay(P0, rate, dt, nsteps):
    """The exact solution of the backward-Euler discretisation of dP/dt = -rate*P.

    Asserting against this rather than exp(-rate*t) isolates the enclosure coupling
    from the time discretisation error.
    """
    return P0 * (1 + rate * dt) ** (-nsteps)


@pytest.mark.parametrize("n", [8, 16, 32, 128])
def test_pump_exponential_decay(n):
    """A closed enclosure pumped at speed S decays as P(t) = P0*exp(-S*t/V).

    Parameterised over mesh sizes: the pressure lives in a real function space, whose
    single global dof is independent of the mesh, so the answer must be too.
    """
    P0, S, V = 1e5, 1e-4, 1e-3
    dt, final_time = 1.0, 20.0

    H2 = F.GasSpecies(name="H2", initial_pressure=P0)
    enclosure = F.Enclosure(
        volume=V, species=[H2], temperature=500.0, openings=[F.Pump(pumping_speed=S)]
    )
    my_model, *_ = make_model(n=n, enclosures=[enclosure], final_time=final_time, dt=dt)
    my_model.initialise()
    my_model.run()

    nsteps = round(final_time / dt)
    expected = backward_euler_decay(P0, S / V, dt, nsteps)
    assert H2.value == pytest.approx(expected, rel=1e-8)


def test_reservoir_relaxes_to_external_pressure():
    """An enclosure connected to a reservoir relaxes as
    P(t) = P_ext + (P0 - P_ext)*exp(-C*t/V)."""
    P0, P_ext, C, V = 1e5, 1e3, 1e-4, 1e-3
    dt, final_time = 1.0, 20.0

    H2 = F.GasSpecies(name="H2", initial_pressure=P0)
    enclosure = F.Enclosure(
        volume=V,
        species=[H2],
        temperature=500.0,
        openings=[F.Reservoir(conductance=C, pressure=P_ext)],
    )
    my_model, *_ = make_model(enclosures=[enclosure], final_time=final_time, dt=dt)
    my_model.initialise()
    my_model.run()

    nsteps = round(final_time / dt)
    # same backward-Euler solution, shifted by the reservoir pressure
    expected = P_ext + backward_euler_decay(P0 - P_ext, C / V, dt, nsteps)
    assert H2.value == pytest.approx(expected, rel=1e-8)


def test_prescribed_flow_rate_is_linear():
    """A constant flow rate Q into a closed enclosure gives P(t) = P0 + Q*k*T*t/V,
    which backward Euler integrates exactly."""
    P0, Q, V, T = 1e3, 1e18, 1e-3, 500.0
    final_time = 20.0

    H2 = F.GasSpecies(name="H2", initial_pressure=P0)
    enclosure = F.Enclosure(
        volume=V,
        species=[H2],
        temperature=T,
        openings=[F.PrescribedFlowRate(flow_rate=Q)],
    )
    my_model, *_ = make_model(enclosures=[enclosure], final_time=final_time, dt=1.0)
    my_model.initialise()
    my_model.run()

    expected = P0 + Q * F.k_B_SI * T * final_time / V
    assert H2.value == pytest.approx(expected, rel=1e-8)


def test_time_dependent_pumping_speed():
    """A pumping speed given as a callable of t is updated during the time loop."""
    P0, V = 1e5, 1e-3
    dt, final_time = 1.0, 5.0

    H2 = F.GasSpecies(name="H2", initial_pressure=P0)
    enclosure = F.Enclosure(
        volume=V,
        species=[H2],
        temperature=500.0,
        openings=[F.Pump(pumping_speed=lambda t: 1e-5 * t)],
    )
    my_model, *_ = make_model(enclosures=[enclosure], final_time=final_time, dt=dt)
    my_model.initialise()
    my_model.run()

    # backward Euler with a speed re-evaluated at each new time
    expected = P0
    for step in range(1, round(final_time / dt) + 1):
        expected /= 1 + (1e-5 * step * dt) / V * dt
    assert H2.value == pytest.approx(expected, rel=1e-8)


def test_enclosure_connection_two_boxes():
    """Two connected enclosures equalise: their difference decays as
    exp(-C*(1/V1 + 1/V2)*t) and the volume-weighted mean is conserved.

    This exercises the off-diagonal pressure-pressure Jacobian block.
    """
    P1_0, P2_0 = 1e5, 0.0
    V1, V2, C = 1e-3, 2e-3, 1e-4
    dt, final_time = 1.0, 20.0

    H2_a = F.GasSpecies(name="H2_a", initial_pressure=P1_0)
    H2_b = F.GasSpecies(name="H2_b", initial_pressure=P2_0)
    connection = F.EnclosureConnection(conductance=C, species=(H2_a, H2_b))
    enclosure_a = F.Enclosure(
        volume=V1, species=[H2_a], temperature=500.0, openings=[connection]
    )
    # deliberately declared only on side a: the mirror is added automatically
    enclosure_b = F.Enclosure(volume=V2, species=[H2_b], temperature=500.0)

    my_model, *_ = make_model(
        enclosures=[enclosure_a, enclosure_b], final_time=final_time, dt=dt
    )
    my_model.initialise()

    # the pressure-pressure coupling block must exist
    nb_volumes = len(my_model.volume_subdomains)
    assert my_model.J[nb_volumes][nb_volumes + 1] is not None

    my_model.run()

    P1, P2 = H2_a.value, H2_b.value

    nsteps = round(final_time / dt)
    # the difference obeys d(P1-P2)/dt = -C*(1/V1 + 1/V2)*(P1-P2)
    expected_diff = backward_euler_decay(P1_0 - P2_0, C * (1 / V1 + 1 / V2), dt, nsteps)
    assert P1 - P2 == pytest.approx(expected_diff, rel=1e-8)

    # particles are conserved, so the volume-weighted mean does not move
    assert P1 * V1 + P2 * V2 == pytest.approx(P1_0 * V1 + P2_0 * V2, rel=1e-8)


@pytest.mark.parametrize("length, area", [(1.0, 1.0), (2.0, 0.25)])
def test_closed_enclosure_conserves_particles(length, area):
    """A closed enclosure exchanging with a slab through a surface reaction 2H <-> H2
    must conserve hydrogen atoms exactly at every timestep.

    Nothing leaves the system, so ``A*int(c_H) dx + 2*P*V/(k*T)`` is invariant, where
    ``A`` is the area of the membrane facing the enclosure (in 1D the model is per unit
    area, so the solid inventory has to be scaled by it). This is exact (to Newton
    tolerance) rather than approximate, because the same UFL expression drives both the
    species residual and the pressure residual. It is the sharpest check available on
    the sign conventions, on the factor of 2 from the diatomic stoichiometry, and on
    the area factor.

    ``(1.0, 1.0)`` is deliberately not the only case. The pressure test function is a
    global constant, so a term spread over a region assembles to ``|region| * f`` and
    has to be divided by that measure, and the flux has to be multiplied by the physical
    area. Both factors are exactly 1 in the unit case, so a unit-only test would pass
    even with the normalisation and the area dropped entirely.
    """
    V_enc, T, P0 = 1e-3, 500.0, 1e5
    k_d0, k_r0 = 1e15, 1e-25
    dt, final_time = 5.0, 200.0

    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 40, [0.0, length])
    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh=mesh)
    material = F.Material(name="mat", D_0=1e-6, E_D=0.0)
    volume = F.VolumeSubdomain1D(id=1, borders=[0.0, length], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=length)
    my_model.subdomains = [volume, left, right]
    H = F.Species("H", subdomains=[volume])
    my_model.species = [H]
    my_model.temperature = T

    H2 = F.GasSpecies(name="H2", initial_pressure=P0)
    my_model.enclosures = [
        F.Enclosure(volume=V_enc, species=[H2], temperature=T, surfaces={right: area})
    ]
    my_model.boundary_conditions = [
        F.SurfaceReactionBC(
            reactant=[H, H],
            gas_pressure=H2,
            k_r0=k_r0,
            E_kr=0.0,
            k_d0=k_d0,
            E_kd=0.0,
            subdomain=right,
        )
    ]
    my_model.initial_conditions = [
        F.InitialConcentration(value=0.0, species=H, volume=volume)
    ]
    my_model.settings = F.Settings(
        atol=1e-8,
        rtol=1e-10,
        transient=True,
        final_time=final_time,
        stepsize=F.Stepsize(dt),
    )
    my_model.show_progress_bar = False
    my_model.initialise()

    def total_hydrogen_atoms():
        c = H.subdomain_to_post_processing_solution[volume]
        # the 1D model is per unit area, so scale the inventory by the membrane area
        in_solid = area * mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(c * ufl.dx)), op=MPI.SUM
        )
        # 2 atoms per H2 molecule
        in_gas = 2.0 * H2.value * V_enc / (F.k_B_SI * T)
        return in_solid + in_gas

    my_model.post_processing()
    initial_inventory = total_hydrogen_atoms()

    while my_model.t.value < final_time:
        my_model.iterate()
        assert total_hydrogen_atoms() == pytest.approx(initial_inventory, rel=1e-12)

    # the gas must actually have been absorbed, otherwise the test is vacuous
    assert H2.value < P0


def test_enclosure_on_2d_mesh_conserves_particles():
    """A closed enclosure on a 2D mesh, facing one wall through a surface reaction.

    This is the 2D counterpart of the conservation test. It matters because a 2D contact
    surface is a line with real extent (unlike 1D, where a facet is a point of measure
    1), so both the domain-measure normalisation and the out-of-plane depth are
    genuinely non-trivial here.

    The wall has length ``Ly = 2`` (so ``|Gamma| = 2``, not 1), and the enclosure area
    is the out-of-plane depth ``0.5`` (not 1). Conservation is
    ``depth * int(c) dx + 2*P*V/(k*T)``. Parameters are kept well scaled so Newton
    converges from the cold start (a stiff cold start is a solver issue independent of
    the enclosure; see the discontinuous solver's behaviour with a zero initial field).
    """
    Lx, Ly, depth = 1.0, 2.0, 0.5
    V_enc, T = 3.0, 500.0
    c0 = 1.0
    k_d0, k_r0 = 1e-3, 1e-3
    dt, final_time = 0.5, 200.0

    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([Lx, Ly])],
        [24, 24],
        dolfinx.mesh.CellType.triangle,
    )
    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh=mesh)
    material = F.Material(name="mat", D_0=1.0, E_D=0.0)
    volume = F.VolumeSubdomain(id=1, material=material)
    right = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], Lx))
    my_model.subdomains = [volume, right]
    H = F.Species("H", subdomains=[volume])
    my_model.species = [H]
    my_model.temperature = T

    H2 = F.GasSpecies(name="H2", initial_pressure=0.0)
    my_model.enclosures = [
        F.Enclosure(volume=V_enc, species=[H2], temperature=T, surfaces={right: depth})
    ]
    my_model.boundary_conditions = [
        F.SurfaceReactionBC(
            reactant=[H, H],
            gas_pressure=H2,
            k_r0=k_r0,
            E_kr=0.0,
            k_d0=k_d0,
            E_kd=0.0,
            subdomain=right,
        )
    ]
    my_model.initial_conditions = [
        F.InitialConcentration(value=c0, species=H, volume=volume)
    ]
    my_model.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=True,
        final_time=final_time,
        stepsize=F.Stepsize(dt),
    )
    my_model.show_progress_bar = False
    my_model.initialise()

    # |Gamma| is the length of the wall, not 1
    assert my_model.enclosures[0]._contact_measure == pytest.approx(Ly)

    def total_hydrogen_atoms():
        c = H.subdomain_to_post_processing_solution[volume]
        in_solid = depth * mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(c * ufl.dx(domain=volume.submesh))
            ),
            op=MPI.SUM,
        )
        in_gas = 2.0 * H2.value * V_enc / (F.k_B_SI * T)
        return in_solid + in_gas

    # the slab starts uniformly loaded and the gas empty
    initial_inventory = depth * c0 * (Lx * Ly)

    while my_model.t.value < final_time:
        my_model.iterate()
        assert total_hydrogen_atoms() == pytest.approx(initial_inventory, rel=1e-10)

    # a meaningful fraction must have crossed into the gas, else the test is vacuous
    in_gas = 2.0 * H2.value * V_enc / (F.k_B_SI * T)
    assert in_gas / initial_inventory > 0.1


def test_steady_state_closed_enclosure_reaches_surface_equilibrium():
    """A closed enclosure at steady state is determined by its contact surface alone.

    Steady state forces zero net flux through the surface, so the surface reaction is at
    equilibrium: kd*P = kr*c^2. With c fixed at the left of the slab and no other flux,
    c is uniform, giving the analytic P = kr*c0^2/kd. No opening is needed.
    """
    T, c0 = 500.0, 1e20
    k_d0, k_r0 = 1e15, 1e-25

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 20)
    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh=mesh)
    material = F.Material(name="mat", D_0=1e-6, E_D=0.0)
    volume = F.VolumeSubdomain1D(id=1, borders=[0.0, 1.0], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=1.0)
    my_model.subdomains = [volume, left, right]
    H = F.Species("H", subdomains=[volume])
    my_model.species = [H]
    my_model.temperature = T

    H2 = F.GasSpecies(name="H2", initial_pressure=1e5)
    my_model.enclosures = [
        # the area does not affect the steady-state equilibrium: zero net flux forces
        # kd*P = kr*c^2 whatever the area scaling
        F.Enclosure(volume=1e-3, species=[H2], temperature=T, surfaces={right: 0.5})
    ]
    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left, value=c0, species=H),
        F.SurfaceReactionBC(
            reactant=[H, H],
            gas_pressure=H2,
            k_r0=k_r0,
            E_kr=0.0,
            k_d0=k_d0,
            E_kd=0.0,
            subdomain=right,
        ),
    ]
    my_model.settings = F.Settings(atol=1e-8, rtol=1e-10, transient=False)
    my_model.show_progress_bar = False
    my_model.initialise()
    my_model.run()

    assert H2.value == pytest.approx(k_r0 * c0**2 / k_d0, rel=1e-8)


def test_enclosure_in_contact_with_two_materials():
    """One enclosure touching both ends of a two-material slab.

    The two contact surfaces sit on different submeshes, so the single pressure form has
    to pull each surface's concentration from a different submesh via ``entity_maps``.
    This also exercises the multi-surface paths: the contact measure is a sum over
    surfaces, and the scalar terms are spread across both of them.

    Conservation is checked over both materials at once, so it fails if either surface
    is dropped from the balance.
    """
    area = 0.25
    V_enc, T, P0 = 1e-3, 500.0, 1e5
    k_d0, k_r0 = 1e15, 1e-25
    dt, final_time = 5.0, 200.0

    vertices = np.concatenate((np.linspace(0, 0.5, 30), np.linspace(0.5, 1.0, 30)))
    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh1D(vertices)
    material_left = F.Material(D_0=1e-6, E_D=0.0, K_S_0=1, E_K_S=0)
    material_right = F.Material(D_0=2e-6, E_D=0.0, K_S_0=1, E_K_S=0)
    vol_left = F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=material_left)
    vol_right = F.VolumeSubdomain1D(id=2, borders=[0.5, 1.0], material=material_right)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)
    my_model.subdomains = [vol_left, vol_right, left, right]
    my_model.interfaces = [
        F.Interface(id=3, subdomains=[vol_left, vol_right], penalty_term=1000)
    ]
    H = F.Species("H", subdomains=[vol_left, vol_right])
    my_model.species = [H]
    my_model.temperature = T

    H2 = F.GasSpecies(name="H2", initial_pressure=P0)
    # left belongs to vol_left's submesh, right to vol_right's
    my_model.enclosures = [
        F.Enclosure(
            volume=V_enc,
            species=[H2],
            temperature=T,
            surfaces={left: area, right: area},
        )
    ]
    my_model.boundary_conditions = [
        F.SurfaceReactionBC(
            reactant=[H, H],
            gas_pressure=H2,
            k_r0=k_r0,
            E_kr=0.0,
            k_d0=k_d0,
            E_kd=0.0,
            subdomain=surface,
        )
        for surface in (left, right)
    ]
    my_model.initial_conditions = [
        F.InitialConcentration(value=0.0, species=H, volume=volume)
        for volume in (vol_left, vol_right)
    ]
    my_model.settings = F.Settings(
        atol=1e-8,
        rtol=1e-10,
        transient=True,
        final_time=final_time,
        stepsize=F.Stepsize(dt),
    )
    my_model.show_progress_bar = False
    my_model.initialise()

    # two point facets in 1D
    assert my_model.enclosures[0]._contact_measure == pytest.approx(2.0)

    def total_hydrogen_atoms():
        in_solid = 0.0
        for volume in (vol_left, vol_right):
            c = H.subdomain_to_post_processing_solution[volume]
            in_solid += my_model.mesh.mesh.comm.allreduce(
                dolfinx.fem.assemble_scalar(
                    dolfinx.fem.form(c * ufl.dx(domain=volume.submesh))
                ),
                op=MPI.SUM,
            )
        return area * in_solid + 2.0 * H2.value * V_enc / (F.k_B_SI * T)

    my_model.post_processing()
    initial_inventory = total_hydrogen_atoms()

    while my_model.t.value < final_time:
        my_model.iterate()
        assert total_hydrogen_atoms() == pytest.approx(initial_inventory, rel=1e-12)

    assert H2.value < P0


def test_gas_pressure_export():
    """The GasPressure export records the pressure over time."""
    P0, S, V = 1e5, 1e-4, 1e-3
    H2 = F.GasSpecies(name="H2", initial_pressure=P0)
    enclosure = F.Enclosure(
        volume=V, species=[H2], temperature=500.0, openings=[F.Pump(pumping_speed=S)]
    )
    my_model, *_ = make_model(enclosures=[enclosure], final_time=5.0, dt=1.0)
    export = F.GasPressure(field=H2)
    my_model.exports = [export]
    my_model.initialise()
    my_model.run()

    assert len(export.data) == len(export.t) == 5
    assert export.data == sorted(export.data, reverse=True)  # pumped down
    assert export.data[-1] == pytest.approx(H2.value)
