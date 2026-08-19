"""The penalty interface must transmit flux *down* the chemical-potential gradient.

``Interface.penalty_method`` couples the two sides of an interface through a term
proportional to ``penalty_term * (u_1/K_1 - u_0/K_0)``. With the residual convention
``F += D grad(u).grad(v) dx``, the sign of that term decides whether the interface is
a resistance (particles flow from high chemical potential to low, the potential
dropping by J/gamma) or an anti-diffusive element (particles flow up the gradient,
the potential *rising* by J/gamma in the flux direction and the coupled form losing
definiteness).

The wrong sign is invisible to tests without a flux through the interface: an MMS
built as ``c_1 = K_1/K_0**2 * c_0**2`` satisfies flux continuity only with zero
interface flux, so the jump ``+-J/gamma`` vanishes either way. It is also invisible
to any flux-carrying test with tolerances coarser than J/gamma, because the two signs
give jumps of the same magnitude. These tests therefore drive one steady permeation
flux through the interface and check the jump's sign, size and gamma-dependence
against reference values.

Setup: unit square, bottom slab [0, 1/2] (D=5, K_S=6) and top slab [1/2, 1]
(D=2, K_S=3), fixed concentrations on the outer walls at chemical potentials 1 and 0
(``c = K_S`` gives ``mu = 1`` under both Henry and Sievert, so the wall recipe is
law-independent). The slabs carry a flux ``J = D dc/dy`` that is linear in
concentration whatever the law, so with ``R_bot = 1/(2*5*6)`` and
``R_top = 1/(2*2*3)`` the interface state solves the scalar balance::

    mu_upstream(1 - J*R_up) - mu_downstream(J*R_dn) = J/gamma

with ``mu(r) = r`` (Henry) or ``r**2`` (Sievert) on each side. For Henry-Henry this
is ``J = 10*gamma/(gamma + 10)`` and ``jump = 10/(gamma + 10)``. The exact solution
is piecewise linear, hence inside the P1 space: computed values must match the
reference to solver tolerance at any resolution. A flipped sign solves the balance
with ``-J/gamma`` instead -- reversed flux for small gamma, a singular system where
the bulk resistance equals 1/gamma, and a wrong-signed jump beyond -- so every
assertion below fails loudly under it.

Every combination the penalty formulation branches over is covered: the four
{Henry, Sievert} law pairs, both flux directions, and both orders of the interface's
subdomain pair (which drive the "+"/"-" restriction wiring). ``penalty_method`` has
two branches: when the two laws *match* it couples ``c/K`` linearly on both sides --
which enforces the same continuity constraint, since ``(c_0/K_0)**2 = (c_1/K_1)**2``
is equivalent to ``c_0/K_0 = c_1/K_1`` for non-negative concentrations, while keeping
the coupling linear -- and only when they differ does each side get its own law's
expression. The reference balance below mirrors that branch exactly: equal-law pairs
(including Sievert-Sievert and NONE-NONE) follow the linear Henry-Henry algebra, and
only the mixed pairs use ``mu``. One consequence worth knowing: at finite gamma the
transmitted flux per unit ``equality`` differs between the linear and squared drives,
so ``penalty_term`` values are not comparable across law pairs (the continuity limit
gamma -> infinity is unaffected). Materials with ``solubility_law = "none"`` also
take the equal-law branch and are pinned in a separate test; a NONE law mixed with
any other is rejected by ``penalty_method`` itself.

The Nitsche method is deliberately out of scope: its consistency terms currently
omit the diffusion coefficient, so it does not meet these heterogeneous-material
references and needs a test of its own once its formulation is settled.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
from scipy.optimize import brentq

import festim as F

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


def analytic_interface_state(gamma, fed, laws):
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
            - flux / gamma
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

    Feeding from the bottom at gamma = 40, the series balance reduces to (in order
    of ``LAW_PAIRS``): J = 10*40/50 = 8 for Henry-Henry, J**2 + 6J - 144 = 0 for
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
    """The jump equals +J/gamma with the potential dropping in the flux direction.

    Checked for every solubility-law pair, both wall orientations and both orders of
    the interface's subdomain pair, so neither the ``equality`` branch taken, nor the
    flux direction, nor the "+"/"-" restriction wiring can hide a sign error. For
    Henry-Henry fed from the bottom at gamma = 40 the values are (26/5, 2); a flipped
    sign gives (14/3, 10/3) instead, with the downstream potential *above* the fed
    wall's.
    """
    gamma = 40.0
    J, expected_bot, expected_top = analytic_interface_state(gamma, fed, laws)

    c_bot, c_top = run_and_measure(
        penalty=gamma, fed=fed, swap_interface_order=swap_interface_order, laws=laws
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
    # direction by exactly J/gamma...
    assert pot_upstream > pot_downstream
    assert np.isclose(pot_upstream - pot_downstream, J / gamma, atol=1e-8)
    # ...and obeys the maximum principle (no interior potential above the fed wall)
    assert max(pot_bot, pot_top) <= 1 + 1e-8


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_penalty_jump_decreases_monotonically_with_penalty():
    """Raising the penalty must shrink the jump as 10/(gamma + 10) -- monotonically.

    A flipped sign gives -10/(gamma - 10): rising in magnitude towards a singular
    system at gamma = 1/R = 10 (excluded from the sweep on purpose) and arriving
    with the wrong sign beyond it, so both the values and the monotonicity fail.
    """
    jumps = []
    for gamma in (4.0, 8.0, 16.0, 32.0, 64.0):
        c_bot, c_top = run_and_measure(penalty=gamma, fed="bottom")
        jump = c_bot / K_BOT - c_top / K_TOP
        assert np.isclose(jump, 10 / (gamma + 10), atol=1e-8)
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
