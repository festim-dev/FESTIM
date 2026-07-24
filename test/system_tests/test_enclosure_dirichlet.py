"""Enclosures coupled to the solid through a weakly enforced Dirichlet BC (Henry's or
Sieverts' law), as opposed to the surface-reaction (flux) path in
``test_enclosures.py``.

The pressure is an unknown living in a real function space, so it cannot be interpolated
into a ``fem.Function``: these boundary conditions can only be enforced weakly, with
Nitsche. The flux that feeds the pressure balance is then the *numerical* flux
``-D grad(c).n + alpha*D/h*(c - g)``, not the raw gradient, which is what makes the
discrete particle balance exact rather than approximate.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import scipy.optimize
import ufl

import festim as F

from ..markers import requires_dolfinx_011

pytestmark = requires_dolfinx_011


def build_slab_with_enclosure(
    length,
    area,
    volume,
    n=60,
    D_0=1e-1,
    T=500.0,
    P0=1e5,
    K_0=1e15,
    law="henry",
    penalty=100,
    sink_on_far_side=False,
    final_time=2000.0,
    dt=10.0,
):
    """A 1D slab whose left surface exchanges with a gas enclosure through Henry's or
    Sieverts' law, optionally with a perfect sink on the right surface.

    Returns:
        the model, the solid species, the gas species, the volume and the enclosure
    """
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, n, [0.0, length])
    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh=mesh)

    material = F.Material(name="mat", D_0=D_0, E_D=0.0)
    vol = F.VolumeSubdomain1D(id=1, borders=[0.0, length], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=length)
    my_model.subdomains = [vol, left, right]

    H = F.Species("H", subdomains=[vol])
    my_model.species = [H]
    my_model.temperature = T

    H2 = F.GasSpecies(name="H2", initial_pressure=P0)
    enclosure = F.Enclosure(
        volume=volume, species=[H2], temperature=T, surfaces={left: area}
    )
    my_model.enclosures = [enclosure]

    if law == "henry":
        coupling_bc = F.HenrysBC(
            subdomain=left,
            H_0=K_0,
            E_H=0.0,
            pressure=H2,
            species=H,
            enforce_weakly=True,
            penalty=penalty,
        )
    else:
        coupling_bc = F.SievertsBC(
            subdomain=left,
            S_0=K_0,
            E_S=0.0,
            pressure=H2,
            species=H,
            enforce_weakly=True,
            penalty=penalty,
        )
    my_model.boundary_conditions = [coupling_bc]

    if sink_on_far_side:
        my_model.boundary_conditions.append(
            F.FixedConcentrationBC(subdomain=right, value=0.0, species=H)
        )

    my_model.initial_conditions = [
        F.InitialConcentration(value=0.0, species=H, volume=vol)
    ]
    # Both tolerances have to match the magnitude of the problem, which is not specific
    # to this coupling: with concentrations of order 1e20 the residual bottoms out
    # around 1e6 in double precision. FESTIM's rtol is measured against the initial
    # residual of the timestep, so as the solution approaches steady state that initial
    # residual falls towards the same floor and no relative criterion can be met. atol
    # is the absolute floor that ends the step, and it has to sit above the roundoff
    # level: a value of 1e-8 assumes residuals of order 1 and is unreachable here.
    my_model.settings = F.Settings(
        atol=1e8,
        rtol=1e-8,
        transient=True,
        final_time=final_time,
        stepsize=F.Stepsize(dt),
    )
    my_model.show_progress_bar = False
    my_model.initialise()

    return my_model, H, H2, vol, enclosure


def solid_inventory(model, species, volume, area):
    """The number of particles of ``species`` in the solid. The 1D model is per unit
    area, so the inventory is scaled by the area of the membrane."""
    model.post_processing()
    c = species.subdomain_to_post_processing_solution[volume]
    return area * model.mesh.mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(c * ufl.dx)), op=MPI.SUM
    )


@pytest.mark.parametrize("length, area", [(1.0, 1.0), (2.0, 0.25)])
@pytest.mark.parametrize("law, stoichiometry", [("henry", 1), ("sieverts", 2)])
def test_dirichlet_coupled_enclosure_conserves_particles(
    length, area, law, stoichiometry
):
    """A closed enclosure exchanging with a slab through a weakly enforced Henry's or
    Sieverts' BC must conserve hydrogen atoms at every timestep.

    Nothing leaves the system, so ``A*int(c) dx + s*P*V/(k*T)`` is invariant, with ``s``
    the number of solid particles per gas molecule (1 for Henry, 2 for Sieverts).

    This is the sharpest check on the whole Dirichlet path at once: the sign of the
    flux, the stoichiometry, the area factor, and above all the use of the *numerical*
    flux. Feeding the raw ``-D grad(c).n`` into the pressure balance instead leaves an
    O(h^p) mismatch between what the solid loses and what the gas gains, which
    accumulates in exactly the quantity being checked here.

    As in the surface-reaction tests, the non-unit parameterisation is what gives the
    test teeth: both the area factor and the normalisation are exactly 1 when
    ``length == area == 1``.
    """
    V_enc, T = 1e-6, 500.0
    model, H, H2, vol, _ = build_slab_with_enclosure(
        length=length, area=area, volume=V_enc, law=law, final_time=200.0, dt=10.0
    )

    def total_hydrogen_atoms():
        in_gas = stoichiometry * H2.value * V_enc / (F.k_B_SI * T)
        return solid_inventory(model, H, vol, area) + in_gas

    initial_inventory = total_hydrogen_atoms()

    while model.t.value < model.settings.final_time:
        model.iterate()
        assert total_hydrogen_atoms() == pytest.approx(initial_inventory, rel=1e-10)


def test_henry_coupled_enclosure_reaches_analytical_equilibrium():
    """A closed enclosure coupled by Henry's law settles at the pressure that shares the
    initial inventory between the gas and a slab in equilibrium with it.

    At equilibrium ``c = K_H * P`` everywhere, so
    ``A*l*K_H*P_inf + P_inf*V/(k*T) == P0*V/(k*T)``. Unlike the conservation test, this
    pins the absolute magnitude of the coupling: the equilibrium moves if ``K_H``,
    the area, or the gas constant is wrong.
    """
    length, area, V_enc, T, P0, K_H = 2.0, 0.25, 1e-6, 500.0, 1e5, 1e15
    model, _, H2, _, _ = build_slab_with_enclosure(
        length=length,
        area=area,
        volume=V_enc,
        law="henry",
        P0=P0,
        K_0=K_H,
        final_time=4000.0,
        dt=20.0,
    )

    while model.t.value < model.settings.final_time:
        model.iterate()

    kT = F.k_B_SI * T
    expected = P0 / (1 + area * length * K_H * kT / V_enc)

    assert H2.value == pytest.approx(expected, rel=1e-4)


def test_tmap_verification_decay_rate():
    """TMAP verification case of issue #996: a slab with a perfect sink on one side and
    an enclosure coupled by Henry's law on the other.

    Separating variables with ``c(l,t) = 0`` and the dynamic boundary condition
    ``V/(k*T*K_H) dc/dt(0,t) = A*D*dc/dx(0,t)`` gives the eigenvalue equation

        ``lambda * tan(lambda) = beta``,   ``beta = k*T*K_H*A*l/V``

    so at late times the pressure decays as ``exp(-D*lambda_1**2*t/l**2)``.

    ``beta`` is dimensionless only with Boltzmann's constant in SI (``F.k_B_SI``): using
    ``F.R`` (J/mol/K) would be wrong by Avogadro's number, so this is an independent
    confirmation of that choice. The decay rate depends on every term of the coupling at
    once, so a sign error, a missing stoichiometry or a mis-scaled flux all fail it.

    The rate is compared to the *discrete* backward-Euler rate rather than the
    continuous one. Integrating ``dP/dt = -r*P`` with backward Euler gives
    ``P_{n+1} = P_n/(1 + r*dt)``, ie. a log-slope of ``-ln(1 + r*dt)/dt``, which differs
    from ``-r`` by an O(dt) amount that has nothing to do with the coupling being
    verified. Both are checked: each timestep matches its own discrete rate tightly, and
    refining dt moves the measured rate towards the continuous one.
    """
    length, area, V_enc, T, P0, K_H, D_0 = 2.0, 0.25, 1e-6, 500.0, 1e5, 1e15, 1e-1
    final_time = 300.0

    beta = F.k_B_SI * T * K_H * area * length / V_enc
    lambda_1 = scipy.optimize.brentq(
        lambda lam: lam * np.tan(lam) - beta, 1e-12, np.pi / 2 - 1e-9
    )
    continuous_rate = D_0 * lambda_1**2 / length**2

    errors_vs_continuous = []
    for dt in (2.0, 0.5):
        model, _, H2, _, _ = build_slab_with_enclosure(
            length=length,
            area=area,
            volume=V_enc,
            D_0=D_0,
            P0=P0,
            K_0=K_H,
            law="henry",
            sink_on_far_side=True,
            final_time=final_time,
            dt=dt,
        )

        times, pressures = [], []
        while model.t.value < final_time:
            model.iterate()
            times.append(float(model.t))
            pressures.append(H2.value)

        times = np.array(times)
        pressures = np.array(pressures)

        # fit where the first eigenmode dominates: after the higher modes have died out
        # but before the pressure gets so small that the solve stalls on its tolerances
        window = (pressures < 1e-2 * pressures[0]) & (pressures > 1e-6 * pressures[0])
        assert window.sum() > 20, "not enough points to fit the decay rate"
        fitted_rate = -np.polyfit(times[window], np.log(pressures[window]), 1)[0]

        discrete_rate = np.log(1 + continuous_rate * dt) / dt
        assert fitted_rate == pytest.approx(discrete_rate, rel=2e-3)

        errors_vs_continuous.append(abs(fitted_rate - continuous_rate))

    # refining the timestep must move the measured rate towards the continuous one
    assert errors_vs_continuous[1] < 0.5 * errors_vs_continuous[0]


def test_time_dependent_temperature_leaves_coupled_bc_untouched():
    """A time-dependent temperature must not break an enclosure-coupled Dirichlet BC.

    The value of such a BC is a pure ufl expression built on live coefficients (the
    temperature constant and the enclosure pressure), so it already tracks their current
    values and must never be re-interpolated. A time-dependent temperature is the one
    thing that would otherwise route the BC through ``update()``: because the coupled
    value depends on ``T``, the temperature loop of ``update_time_dependent_values``
    calls ``bc.update(t)`` every step. That call has to return early -- trying to
    interpolate a bare ufl expression into a ``fem.Function`` would raise -- which is
    what this test exercises.
    """
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 20, [0.0, 1.0])
    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh=mesh)

    material = F.Material(name="mat", D_0=1e-1, E_D=0.0)
    vol = F.VolumeSubdomain1D(id=1, borders=[0.0, 1.0], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=1.0)
    my_model.subdomains = [vol, left, right]

    H = F.Species("H", subdomains=[vol])
    my_model.species = [H]
    # a constant value, but given as a callable of t so it is flagged time-dependent
    my_model.temperature = lambda t: 500.0

    H2 = F.GasSpecies(name="H2", initial_pressure=1e5)
    my_model.enclosures = [
        F.Enclosure(volume=1e-6, species=[H2], temperature=500.0, surfaces={left: 1.0})
    ]
    coupling_bc = F.HenrysBC(
        subdomain=left,
        H_0=1e15,
        E_H=0.0,
        pressure=H2,
        species=H,
        enforce_weakly=True,
        penalty=100,
    )
    my_model.boundary_conditions = [coupling_bc]
    my_model.initial_conditions = [
        F.InitialConcentration(value=0.0, species=H, volume=vol)
    ]
    my_model.settings = F.Settings(
        atol=1e8, rtol=1e-8, transient=True, final_time=20.0, stepsize=F.Stepsize(10.0)
    )
    my_model.show_progress_bar = False
    my_model.initialise()

    # the value stays a pure ufl expression, not a Function/Constant, before and after
    # a step through the time loop
    assert isinstance(coupling_bc.value_fenics, ufl.core.expr.Expr)
    assert not isinstance(
        coupling_bc.value_fenics, (dolfinx.fem.Function, dolfinx.fem.Constant)
    )
    my_model.iterate()
    assert isinstance(coupling_bc.value_fenics, ufl.core.expr.Expr)
    assert not isinstance(
        coupling_bc.value_fenics, (dolfinx.fem.Function, dolfinx.fem.Constant)
    )


def test_dirichlet_coupling_agrees_with_surface_reaction():
    """The Dirichlet path and the trusted surface-reaction path must agree.

    A surface reaction in the fast-exchange limit is in local equilibrium, so
    ``k_d*P == k_r*c**2``, ie. Sieverts' law with ``S = sqrt(k_d/k_r)``. Driving the
    same physical problem both ways validates the weakly enforced Dirichlet coupling
    against the flux coupling, which is exactly conservative by construction.
    """
    length, area, V_enc, T, P0 = 1.0, 1.0, 1e-6, 500.0, 1e5
    k_d0, k_r0 = 1e13, 1e-21
    S = np.sqrt(k_d0 / k_r0)
    final_time, dt, n = 400.0, 10.0, 60
    D_0 = 1e-1

    # --- Sieverts (weakly enforced Dirichlet)
    model_d, _, H2_d, _, _ = build_slab_with_enclosure(
        length=length,
        area=area,
        volume=V_enc,
        n=n,
        D_0=D_0,
        P0=P0,
        K_0=S,
        law="sieverts",
        final_time=final_time,
        dt=dt,
    )
    while model_d.t.value < final_time:
        model_d.iterate()

    # --- the same problem through a surface reaction
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, n, [0.0, length])
    model_r = F.HydrogenTransportProblemDiscontinuous()
    model_r.mesh = F.Mesh(mesh=mesh)
    material = F.Material(name="mat", D_0=D_0, E_D=0.0)
    vol_r = F.VolumeSubdomain1D(id=1, borders=[0.0, length], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=length)
    model_r.subdomains = [vol_r, left, right]
    H_r = F.Species("H", subdomains=[vol_r])
    model_r.species = [H_r]
    model_r.temperature = T
    H2_r = F.GasSpecies(name="H2", initial_pressure=P0)
    model_r.enclosures = [
        F.Enclosure(volume=V_enc, species=[H2_r], temperature=T, surfaces={left: area})
    ]
    model_r.boundary_conditions = [
        F.SurfaceReactionBC(
            reactant=[H_r, H_r],
            gas_pressure=H2_r,
            k_r0=k_r0,
            E_kr=0.0,
            k_d0=k_d0,
            E_kd=0.0,
            subdomain=left,
        )
    ]
    model_r.initial_conditions = [
        F.InitialConcentration(value=0.0, species=H_r, volume=vol_r)
    ]
    # same reasoning as in build_slab_with_enclosure: atol has to sit above the
    # roundoff floor of a problem whose concentrations are of order 1e19
    model_r.settings = F.Settings(
        atol=1e8,
        rtol=1e-8,
        transient=True,
        final_time=final_time,
        stepsize=F.Stepsize(dt),
    )
    model_r.show_progress_bar = False
    model_r.initialise()
    while model_r.t.value < final_time:
        model_r.iterate()

    # the two are different discretisations of the same physics, compared once both
    # have settled: the fast-exchange limit is an approximation, so this is not expected
    # to be exact. It agrees far better than that in practice (~1e-6).
    assert H2_d.value == pytest.approx(H2_r.value, rel=5e-3)
