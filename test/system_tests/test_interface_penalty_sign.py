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

    J = delta_drive / (R + 1/penalty_term)
    R = L_0/(D_0 K_0) + L_1/(D_1 K_1)

for the correct sign, and R - 1/penalty_term for the wrong one. A
manufactured solution that satisfies the interface condition solves both
forms exactly, so an MMS test on its own detects nothing. What the sign
changes is the gain applied to the residual that remains:
1/(R + 1/penalty_term) is monotone and bounded for every
penalty_term, while 1/(R - 1/penalty_term) is singular at
penalty_term = 1/R and negative below it, i.e. transport runs up the
chemical potential gradient.

The bug therefore hides when penalty_term is well below 1/R *and*
the interface carries no flux, which was the regime of the existing
2 material mms test with penalty_term=1. It appears once either
condition breaks: a real flux through the interface (#1233), or a
penalty_term near 1/R (#1236, where R ~ 0.1 and the MWE uses
penalty_term = 10).

The tests below break both conditions deliberately.
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
