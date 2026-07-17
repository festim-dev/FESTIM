from mpi4py import MPI

import dolfinx
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


@pytest.mark.parametrize("length", [1.0, 2.0])
def test_closed_enclosure_conserves_particles(length):
    """A closed enclosure exchanging with a slab through a surface reaction 2H <-> H2
    must conserve hydrogen atoms exactly at every timestep.

    Nothing leaves the system, so ``int(c_H) dx + 2*P*V/(k*T)`` is invariant. This is
    exact (to Newton tolerance) rather than approximate, because the same UFL
    expression drives both the species residual and the pressure residual. It is the
    sharpest check available on the sign conventions and on the factor of 2 from the
    diatomic stoichiometry.

    ``length`` is parameterised, and 1.0 is deliberately not the only case: the pressure
    test function is a global constant, so a volume term assembles to ``|Omega| * f``
    while a surface term does not pick up that factor. The residual divides its volume
    terms by ``|Omega|`` to compensate, and on a unit domain that division is a no-op --
    so a unit-only test passes even if the normalisation is missing entirely.
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
        F.Enclosure(volume=V_enc, species=[H2], temperature=T, surfaces=[right])
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
        in_solid = mesh.comm.allreduce(
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
        F.Enclosure(volume=1e-3, species=[H2], temperature=T, surfaces=[right])
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
