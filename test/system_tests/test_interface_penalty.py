"""The penalty interface must transmit flux *down* the chemical-potential gradient.

NOTE: Here `penalty_term`` is the paper's alpha scaled by K_0, since here the jump is
normalized by K_0.

The interface couples two subdomains with
penalty_term * (u_0/K_0 - u_1/K_1) * (v_0 - v_1) * dS. With the residual
convention F += D grad(u).grad(v) dx, this sign makes the interface a
resistance in series with the bulk: the chemical potential drops by
J/penalty_term in the direction of the flux. The reversed order
(u_1/K_1 - u_0/K_0), used before #1237, makes it a negative resistance
that cancels against the bulk instead.

For a 1D two-layer problem, eliminating the interface values gives

    J = delta_drive / (R + 1/penalty_term)    [flux]
    R = L_0/(D_0 K_0) + L_1/(D_1 K_1)         [interface resistance]

for the correct sign, and (R - 1/penalty_term) for the wrong one.
What the sign changes is the constitutive relation at the interface:
the correct sign gives a, monotone, bounded value for every penalty_term,
while the reversed one gives a singularity at penalty_term = 1/R and
negative below it, i.e. transport runs up the chemical potential gradient.
An MMS that satisfies the interface condition with zero interface flux solves
both forms exactly, so it detects nothing directly; there the wrong sign shows
up only as ill-conditioning amplifying the discretization residual,
which is why #1236 needed penalty_term near 1/R to surface it. The
tests below instead drive a real flux across the interface, where the
wrong sign is a first-order error in the solution and no conditioning
argument is needed.

The second half of the file pins the two properties that make this penalty a
*model* rather than an approximation: the interface neither creates nor destroys
particles at any finite ``penalty_term``, and the state it converges to as
``penalty_term -> infinity`` is the true continuity limit. Issue #1026 proposes
replacing this form with the gradient of the quadratic functional
``penalty_term/2 * integral(equality**2)``, i.e.
``penalty_term * equality * (deq/du_0 v_0 + deq/du_1 v_1)``; the last two tests
implement that variant and show what it costs -- a conservation error that does
*not* vanish as penalty_term grows, and a spurious converged state in which an
empty Sievert subdomain decouples from the interface entirely.

The last section covers ``nitsche_method`` against the same references. Nitsche adds
the consistency term the penalty lacks -- the transmitted flux ``{D grad(c) . n}``
written explicitly -- so the exact solution satisfies the discrete form for *any*
``penalty_term``. The slab solutions here are piecewise linear, hence in the P1 space,
so a consistent method must return them to solver tolerance even at
``penalty_term = 10``, where the penalty method still carries a visible
1/penalty_term error. That makes these tests sensitive to the two things the
formulation must not drop: the diffusion coefficient inside the flux average, and
the solubility-law branch in :meth:`Interface.equality` (without it a Sievert/Henry
pair is coupled through ``c/K`` instead of partial pressures, and converges to a
different state entirely).
"""

import warnings

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl
from scipy.optimize import brentq

import festim as F
from festim.material import SolubilityLaw
from festim.subdomain.interface import Interface, InterfaceMethod

D_BOT, K_BOT = 5.0, 6.0
D_TOP, K_TOP = 2.0, 3.0
R_BOT, R_TOP = 1 / (2 * D_BOT * K_BOT), 1 / (2 * D_TOP * K_TOP)  # 1/60, 1/12
BOT_ID, TOP_ID, BOTTOM_SURF_ID, TOP_SURF_ID, INTERFACE_ID = 1, 2, 3, 4, 6

LAW_PAIRS = [
    ("henry", "henry"),
    ("henry", "sievert"),
    ("sievert", "henry"),
    ("sievert", "sievert"),
]


def mu(law, ratio):
    """Chemical potential as a function of the ratio c/K_S."""
    return ratio**2 if law == "sievert" else ratio


def coupling_potential(laws, law, ratio):
    """The expression ``penalty_method`` couples on one side: the plain ratio
    ``c/K_S`` when the two laws match (the equal-law branch), the law's own
    chemical potential when they differ."""
    return ratio if laws[0] == laws[1] else mu(law, ratio)


def analytic_interface_state(penalty_term, fed, laws):
    """The exact flux and interface concentrations of the series balance.

    ``g(J)`` below is strictly decreasing on ``[0, 1/R_up]`` with ``g(0) = 1`` and
    ``g(1/R_up) < 0``, so the physical root is unique and bracketed.
    """
    law_bot, law_top = laws
    if fed == "bottom":
        law_up, law_dn, R_up, R_dn = law_bot, law_top, R_BOT, R_TOP
    else:
        law_up, law_dn, R_up, R_dn = law_top, law_bot, R_TOP, R_BOT

    def g(flux):
        return (
            coupling_potential(laws, law_up, 1 - flux * R_up)
            - coupling_potential(laws, law_dn, flux * R_dn)
            - flux / penalty_term
        )

    J = brentq(g, 0.0, 1 / R_up, xtol=1e-14)
    if fed == "bottom":
        return J, K_BOT * (1 - J * R_BOT), K_TOP * J * R_TOP
    return J, K_BOT * J * R_BOT, K_TOP * (1 - J * R_TOP)


def build(penalty, fed="bottom", swap_interface_order=False, laws=None, n=8):
    """Two half-slabs joined by a penalty ``Interface``, walls at mu = 1 and 0.

    ``n`` must be even so the interface lies on a mesh line; the values checked are
    resolution-independent beyond that (the exact solution is in the P1 space).
    """
    laws = laws or ("henry", "henry")
    material_bottom = F.Material(D_0=D_BOT, E_D=0, K_S_0=K_BOT, E_K_S=0)
    material_top = F.Material(D_0=D_TOP, E_D=0, K_S_0=K_TOP, E_K_S=0)
    material_bottom.solubility_law = laws[0]
    material_top.solubility_law = laws[1]

    bottom = F.VolumeSubdomain(
        id=BOT_ID, material=material_bottom, locator=lambda x: x[1] <= 0.5 + 1e-14
    )
    top = F.VolumeSubdomain(
        id=TOP_ID, material=material_top, locator=lambda x: x[1] >= 0.5 - 1e-14
    )
    bottom_surf = F.SurfaceSubdomain(
        id=BOTTOM_SURF_ID, locator=lambda x: np.isclose(x[1], 0.0)
    )
    top_surf = F.SurfaceSubdomain(
        id=TOP_SURF_ID, locator=lambda x: np.isclose(x[1], 1.0)
    )

    H = F.Species("H", subdomains=[bottom, top])

    # c = K_S gives mu = 1 under Henry, Sievert and NONE alike, so the fed wall sits
    # at potential 1 and the other at 0 whatever the law pair
    c_bot_wall, c_top_wall = (K_BOT, 0.0) if fed == "bottom" else (0.0, K_TOP)

    pair = (top, bottom) if swap_interface_order else (bottom, top)
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)),
        species=[H],
        subdomains=[bottom, top, bottom_surf, top_surf],
        boundary_conditions=[
            F.FixedConcentrationBC(bottom_surf, value=c_bot_wall, species=H),
            F.FixedConcentrationBC(top_surf, value=c_top_wall, species=H),
        ],
        interfaces=[F.Interface(INTERFACE_ID, pair, penalty_term=penalty)],
        temperature=500.0,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    # ``model.exports[0]`` is the bottom wall, ``[1]`` the top one; see wall_fluxes
    model.exports = [
        F.SurfaceFlux(field=H, surface=bottom_surf),
        F.SurfaceFlux(field=H, surface=top_surf),
    ]
    return model, (bottom, top), H


def interface_trace(species, subdomain):
    """The (constant) value of ``species`` on ``subdomain``'s side of y = 1/2."""
    f = species.subdomain_to_post_processing_solution[subdomain]
    coords = f.function_space.tabulate_dof_coordinates()
    on_interface = np.isclose(coords[:, 1], 0.5)
    assert on_interface.any(), "no dofs on y = 1/2; the mesh resolution must be even"
    values = f.x.array.real[on_interface]
    # the problem is one-dimensional in y, so the trace must be uniform along x
    assert np.ptp(values) < 1e-8
    return values.mean()


def run_and_measure(**kwargs):
    model, (bottom, top), H = build(**kwargs)
    model.initialise()
    model.run()
    return interface_trace(H, bottom), interface_trace(H, top)


def test_analytic_reference_values():
    """Pin the reference solver itself against hand-derived closed forms.

    Feeding from the bottom at penalty_term = 40, the series balance reduces to (in
    order of ``LAW_PAIRS``): J = 10*40/50 = 8 for Henry-Henry, J**2 + 6J - 144 = 0 for
    Henry-Sievert, J**2 - 510J + 3600 = 0 for Sievert-Henry, and -- because equal
    laws take the linear branch -- J = 8 again for Sievert-Sievert. Feeding a
    Sievert top over a Henry bottom gives J**2 - 30J + 144 = 0, i.e. exactly J = 6.
    """
    hand_derived = [
        8.0,
        np.sqrt(153) - 3,
        255 - 15 * np.sqrt(273),
        8.0,
    ]
    for laws, expected in zip(LAW_PAIRS, hand_derived):
        J, _, _ = analytic_interface_state(40.0, "bottom", laws)
        assert np.isclose(J, expected, atol=1e-12)
    J, c_bot, c_top = analytic_interface_state(40.0, "top", ("henry", "sievert"))
    assert np.isclose(J, 6.0, atol=1e-12)
    assert np.isclose(c_bot, 0.6, atol=1e-12) and np.isclose(c_top, 1.5, atol=1e-12)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
@pytest.mark.parametrize("swap_interface_order", [False, True])
@pytest.mark.parametrize("fed", ["bottom", "top"])
@pytest.mark.parametrize("laws", LAW_PAIRS, ids=lambda p: "-".join(p))
def test_penalty_interface_matches_series_resistance_solution(
    laws, fed, swap_interface_order
):
    """The jump equals +J/penalty_term with the potential dropping in the flux direction.

    Checked for every solubility-law pair, both wall orientations and both orders of
    the interface's subdomain pair, so neither the ``equality`` branch taken, nor the
    flux direction, nor the "+"/"-" restriction wiring can hide a sign error. For
    Henry-Henry fed from the bottom at penalty_term = 40 the values are (26/5, 2); a flipped
    sign gives (14/3, 10/3) instead, with the downstream potential *above* the fed
    wall's.
    """  # noqa: E501
    penalty_term = 40.0
    J, expected_bot, expected_top = analytic_interface_state(penalty_term, fed, laws)

    c_bot, c_top = run_and_measure(
        penalty=penalty_term,
        fed=fed,
        swap_interface_order=swap_interface_order,
        laws=laws,
    )
    # the quantities the coupling actually drives: linear ratios for equal laws,
    # per-law potentials for mixed ones. For equal Sievert sides the ordering of
    # c/K and (c/K)**2 coincides, so the sign checks below lose no generality
    pot_bot = coupling_potential(laws, laws[0], c_bot / K_BOT)
    pot_top = coupling_potential(laws, laws[1], c_top / K_TOP)

    assert np.isclose(c_bot, expected_bot, atol=1e-8)
    assert np.isclose(c_top, expected_top, atol=1e-8)

    # one conserved flux crosses both slabs (this holds under either sign, so it
    # isolates the coupling's sign as the only thing the value checks can trip on)
    if fed == "bottom":
        flux_bot, flux_top = 2 * D_BOT * (K_BOT - c_bot), 2 * D_TOP * c_top
        pot_upstream, pot_downstream = pot_bot, pot_top
    else:
        flux_bot, flux_top = 2 * D_BOT * c_bot, 2 * D_TOP * (K_TOP - c_top)
        pot_upstream, pot_downstream = pot_top, pot_bot
    assert np.isclose(flux_bot, J, atol=1e-7) and np.isclose(flux_top, J, atol=1e-7)

    # the sign itself, stated as physics: the coupled potential drops in the flux
    # direction by exactly J/penalty_term...
    assert pot_upstream > pot_downstream
    assert np.isclose(pot_upstream - pot_downstream, J / penalty_term, atol=1e-8)
    # ...and obeys the maximum principle (no interior potential above the fed wall)
    assert max(pot_bot, pot_top) <= 1 + 1e-8


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_penalty_jump_decreases_monotonically_with_penalty():
    """Raising the penalty must shrink the jump as 10/(penalty_term + 10) monotonically.

    A flipped sign gives -10/(penalty_term - 10): rising in magnitude towards a singular
    system at penalty_term = 1/R = 10 (excluded from the sweep on purpose) and arriving
    with the wrong sign beyond it, so both the values and the monotonicity fail.
    """
    jumps = []
    for penalty_term in (4.0, 8.0, 16.0, 32.0, 64.0):
        c_bot, c_top = run_and_measure(penalty=penalty_term, fed="bottom")
        jump = c_bot / K_BOT - c_top / K_TOP
        assert np.isclose(jump, 10 / (penalty_term + 10), atol=1e-8)
        jumps.append(jump)
    assert np.all(np.diff(jumps) < 0)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_none_solubility_law_interface_behaves_like_henry():
    """Materials without a solubility law take the same equal-law (linear) branch
    as matching Henry or Sievert pairs, so an interface between two of them must
    reproduce the Henry-Henry values. If FESTIM later chooses to reject NONE-law
    materials at interfaces instead, replace this with a ``pytest.raises``.
    """
    c_bot, c_top = run_and_measure(penalty=40.0, fed="bottom", laws=("none", "none"))
    assert np.isclose(c_bot, 26 / 5, atol=1e-8)
    assert np.isclose(c_top, 2.0, atol=1e-8)


def wall_fluxes(model):
    """Flux through the bottom and top walls, after ``run``.

    ``SurfaceFlux`` measures ``-D grad(c) . n`` with ``n`` the outward normal of the
    subdomain, so a steady state with no volumetric source has ``f_bot + f_top == 0``
    whatever the direction of the flow: whatever enters through one wall leaves
    through the other. Nothing else in the model can absorb particles, so this sum is
    a direct measurement of what the interface does to the balance.
    """
    export_bot, export_top = model.exports
    return export_bot.data[-1], export_top.data[-1]


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
@pytest.mark.parametrize("penalty", [10.0, 1e3, 1e5])
@pytest.mark.parametrize("laws", LAW_PAIRS, ids=lambda p: "-".join(p))
def test_penalty_interface_conserves_flux(laws, penalty):
    """The interface transmits the flux it receives -- exactly, at any penalty.

    A penalty interface is an approximation of continuity, so at finite
    ``penalty_term`` the chemical potential does jump (that is the point, and the
    jump is checked above). What must *not* be approximate is the particle balance:
    ``penalty_method`` adds the same term to both sides with opposite signs, so the
    natural condition it imposes is ``D grad(c_0) . n = D grad(c_1) . n`` identically,
    independently of how far ``equality`` is from zero.

    This is the property that separates the implemented form from the quadratic
    penalty of issue #1026 -- see the last test -- so it is checked across the law
    pairs and over four decades of ``penalty_term``, including deliberately loose
    values where ``equality`` is nowhere near zero.
    """
    model, _, _ = build(penalty=penalty, laws=laws)
    model.initialise()
    model.run()

    f_bot, f_top = wall_fluxes(model)
    assert abs(f_bot) > 1  # a real flux crosses the interface, so the check has teeth
    assert np.isclose(f_bot, -f_top, rtol=1e-9)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
@pytest.mark.parametrize("laws", [("sievert", "henry"), ("henry", "sievert")])
def test_penalty_converges_to_the_continuity_limit(laws):
    """Raising the penalty must recover the interface condition it approximates.

    The limit state is the one issue #1026 writes as ``u_A = K_A (u_B/K_B)**n``:
    equal chemical potentials on both sides, plus flux continuity. FESTIM writes the
    same constraint as a pressure balance, ``(c_0/K_0)**2 = c_1/K_1``, which is a
    polynomial rather than a fractional power -- no ``sqrt`` of a possibly negative
    iterate -- and the Newton solver reaches it here for the mixed law pairs the
    issue reports as failing, with the error falling like ``1/penalty_term``.
    """
    _, c_bot_limit, c_top_limit = analytic_interface_state(np.inf, "bottom", laws)

    errors = []
    for penalty in (1e2, 1e3, 1e4):
        c_bot, c_top = run_and_measure(penalty=penalty, laws=laws)
        errors.append(abs(c_bot - c_bot_limit) + abs(c_top - c_top_limit))

    # first order in 1/penalty_term: each decade must buy roughly a decade of error
    assert np.all(np.array(errors[:-1]) / np.array(errors[1:]) > 5)
    assert errors[-1] < 1e-2


def quadratic_penalty_method(self, dS, species, temperature):
    """``d/du`` of ``penalty_term/2 * integral(equality**2)``, as proposed in #1026.

    Same ``equality`` as ``Interface.penalty_method``; only the weighting of the two
    test functions differs -- ``deq/du_0`` and ``deq/du_1`` in place of ``+1`` and
    ``-1``.
    """
    subdomain_0, subdomain_1 = self.subdomains
    u_0, u_1 = self.us(species)
    v_0, v_1 = self.vs(species)
    K_0, K_1 = self.Ks(species, temperature)
    same_law = (
        subdomain_0.material.solubility_law == subdomain_1.material.solubility_law
    )

    def side(subdomain, u, K):
        """A side's term in ``equality``, and its derivative with respect to ``u``."""
        if not same_law and subdomain.material.solubility_law == SolubilityLaw.SIEVERT:
            return (u / K) ** 2, 2 * u / K**2
        return u / K, 1 / K

    left, dleft = side(subdomain_0, u_0, K_0)
    right, dright = side(subdomain_1, u_1, K_1)
    equality = left - right

    F_0 = self.penalty_term * ufl.inner(equality * dleft, v_0) * dS(self.id)
    F_1 = -self.penalty_term * ufl.inner(equality * dright, v_1) * dS(self.id)
    return F_0, F_1


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_quadratic_penalty_variant_breaks_flux_conservation(monkeypatch):
    """Why the interface is *not* formulated as the gradient of a quadratic penalty.

    Minimising ``E(c) + penalty_term/2 * integral(equality**2)`` makes the two natural
    conditions ``D grad(c_0) . n = -lambda deq/du_0`` and
    ``D grad(c_1) . n = +lambda deq/du_1`` with ``lambda = penalty_term * equality``.
    Those agree only when ``deq/du_0 == -deq/du_1``, which for
    ``equality = (c_0/K_0)**2 - c_1/K_1`` means ``2 c_0/K_0**2 == 1/K_1`` -- true at
    best at one point. Everywhere else the interface creates or destroys particles.

    The gap does not close as ``penalty_term`` grows: ``equality`` tends to zero but
    ``lambda`` tends to the transmitted flux, so the mismatch stays a fixed fraction
    of it (~15% here) and the converged state is simply the wrong one. Issue #1026's
    manufactured solution cannot see this, because its parameters
    (``n = 1/2``, ``K_A = 1``, ``K_B = n d**((n-1)/n)``) give ``deq/du_B = -1`` exactly
    at the interface value of its exact solution, where the quadratic form and the
    implemented one coincide.
    """
    monkeypatch.setattr(Interface, "penalty_method", quadratic_penalty_method)

    imbalances = []
    for penalty in (1e3, 1e5):
        model, _, _ = build(penalty=penalty, laws=("sievert", "henry"))
        model.initialise()
        model.run()
        f_bot, f_top = wall_fluxes(model)
        imbalances.append(abs(f_bot + f_top) / abs(f_bot))

    # a real violation, and one that a larger penalty does not repair
    assert min(imbalances) > 0.1
    assert imbalances[1] >= imbalances[0] - 0.01

    # ... and the flux it does transmit is not the one continuity asks for
    J_limit, _, _ = analytic_interface_state(np.inf, "bottom", ("sievert", "henry"))
    model, _, _ = build(penalty=1e5, laws=("sievert", "henry"))
    model.initialise()
    model.run()
    assert not np.isclose(abs(wall_fluxes(model)[0]), J_limit, rtol=0.05)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
@pytest.mark.parametrize("penalty", [1e3, 1e5])
def test_quadratic_penalty_variant_admits_a_spurious_empty_solution(
    monkeypatch, penalty
):
    """The same variant has a second, worse failure: it can decouple the interface.

    Weighting side 0's residual by ``deq/du_0 = 2 c_0/K_0**2`` means the interface
    exerts no force at all on a Sievert material once ``c_0`` reaches zero. An empty
    Sievert subdomain is then a converged solution -- the wall Dirichlet condition
    plus a zero flux at the interface -- and FESTIM's zero initial guess starts on it,
    so the Newton loop reports success on a state that transmits nothing. Here the
    Henry side keeps draining ~12 particles per unit time into an interface that
    delivers none of them.

    The implemented form weights by 1 on both sides, has no such stationary point,
    and returns the series-resistance solution instead.
    """
    monkeypatch.setattr(Interface, "penalty_method", quadratic_penalty_method)

    model, (bottom, _), H = build(penalty=penalty, fed="top", laws=("sievert", "henry"))
    model.initialise()
    model.run()

    f_bot, f_top = wall_fluxes(model)
    assert np.isclose(interface_trace(H, bottom), 0.0, atol=1e-12)  # empty Sievert side
    assert np.isclose(f_bot, 0.0, atol=1e-10)  # transmitting nothing
    assert abs(f_top) > 10  # while the Henry side still empties into it


def run_nitsche(**kwargs):
    """Same as ``run_and_measure`` but with the interface coupled by Nitsche."""
    model, (bottom, top), H = build(**kwargs)
    for interface in model.interfaces:
        interface.method = InterfaceMethod.nitsche
    model.initialise()
    model.run()
    return model, (interface_trace(H, bottom), interface_trace(H, top))


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
@pytest.mark.parametrize("penalty", [10.0, 1e4])
@pytest.mark.parametrize("swap_interface_order", [False, True])
@pytest.mark.parametrize("fed", ["bottom", "top"])
@pytest.mark.parametrize("laws", LAW_PAIRS, ids=lambda p: "-".join(p))
def test_nitsche_reproduces_the_continuity_limit(
    laws, fed, swap_interface_order, penalty
):
    """A consistent method lands on the exact answer, not near it.

    The reference is the ``penalty_term -> infinity`` state: equal chemical
    potentials plus flux continuity. Nitsche has to reach it at ``penalty_term = 10``
    just as well as at 1e4, because the exact solution is piecewise linear -- it lies
    in the P1 space, so consistency alone determines the answer and the stabilisation
    contributes nothing at the solution.

    Dropping ``D`` from the flux average makes the consistency term reproduce
    ``{grad(c) . n}`` instead of the transmitted flux and this fails for every law
    pair (the slabs here have D = 5 and 2); dropping the solubility-law branch fails
    for the mixed pairs only, converging to the ``c/K`` continuity state instead
    (5.0 / 2.5 rather than 5.1246 / 2.1885 for sievert-henry fed from the bottom).
    """
    _, c_bot_limit, c_top_limit = analytic_interface_state(np.inf, fed, laws)

    model, (c_bot, c_top) = run_nitsche(
        penalty=penalty, fed=fed, swap_interface_order=swap_interface_order, laws=laws
    )

    assert np.isclose(c_bot, c_bot_limit, atol=1e-8)
    assert np.isclose(c_top, c_top_limit, atol=1e-8)

    # and, as for the penalty, the interface neither creates nor destroys particles
    f_bot, f_top = wall_fluxes(model)
    assert abs(f_bot) > 1
    assert np.isclose(f_bot, -f_top, rtol=1e-9)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_nitsche_reaches_at_ten_what_the_penalty_needs_thousands_for():
    """The consistency term is what earns the small ``penalty_term``.

    At ``penalty_term = 10`` the pure penalty sits a whole 1/penalty_term error
    away from the continuity limit -- for this problem the interface resistance
    1/penalty_term is itself equal to the bulk resistance, so it halves the flux.
    Nitsche is exact.
    Should the consistency term ever be removed, Nitsche would degrade to exactly the
    penalty's answer and this test would catch it.
    """
    _, c_bot_limit, _ = analytic_interface_state(np.inf, "bottom", LAW_PAIRS[0])

    _, (c_bot_nitsche, _) = run_nitsche(penalty=10.0, laws=LAW_PAIRS[0])
    c_bot_penalty, _ = run_and_measure(penalty=10.0, laws=LAW_PAIRS[0])

    assert np.isclose(c_bot_nitsche, c_bot_limit, atol=1e-8)
    assert abs(c_bot_penalty - c_bot_limit) > 0.1


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_interface_method_survives_initialise():
    """``initialise`` must not overwrite a method set on the interface itself.

    The deprecated problem-level ``method_interface`` used to be pushed onto every
    interface unconditionally -- the guard tested ``hasattr(self,
    "method_interface")``, which is always true for a property -- so the documented
    per-interface attribute was silently reset to the default on every run.
    """
    model, _, _ = build(penalty=10.0)
    model.interfaces[0].method = InterfaceMethod.nitsche
    model.initialise()

    assert model.interfaces[0].method == InterfaceMethod.nitsche


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_no_deprecation_warning_without_the_problem_level_attribute():
    """A model that never touches ``method_interface`` must not be warned about it.

    Filtered by message rather than by category: dolfinx emits its own unrelated
    DeprecationWarnings during ``initialise``.
    """
    model, _, _ = build(penalty=10.0)
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        model.initialise()

    assert not [w for w in raised if "method_interface" in str(w.message)]


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_problem_level_method_interface_still_wins_and_warns():
    """The deprecated route keeps working, warning, and taking precedence."""
    model, _, _ = build(penalty=10.0)
    model.method_interface = "penalty"
    model.interfaces[0].method = InterfaceMethod.nitsche

    with pytest.deprecated_call():
        model.initialise()

    assert model.interfaces[0].method == InterfaceMethod.penalty
