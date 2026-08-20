"""Interface coupling at the magnitudes hydrogen transport actually runs at.

Every other interface test in this directory is written at unit scale -- D, K_S and
concentrations all of order one -- which hides anything that depends on the *units*
of the coupling rather than on its structure. Real problems are nothing like that:
the cases below span ``D ~ 1e-9 m2/s``, solubilities from 1e17 to 1e20,
concentrations up to 1e25 H/m3 and permeation fluxes around 1e14 H/m2/s -- and the
Sievert and Henry coefficients do not even share units.

Two properties are pinned here.

``penalty_term`` is a **surface conductance** for :meth:`Interface.penalty_method`:
the interface imposes ``flux = penalty_term * equality``, so it acts as a contact
resistance ``1/penalty_term`` in series with the bulk ``R = L/(D K)``, and only the
dimensionless product ``penalty_term * R`` matters. With this data ``1/R`` is of
order 1e12, so the default ``penalty_term = 10`` transmits a vanishing fraction of
the correct flux -- a perfect barrier, silently. That is what a pure penalty does,
not a bug, but it has to be stated somewhere a reader will find it.

:meth:`Interface.nitsche_method` must have no such dependence. Its stabilisation is
scaled by :meth:`Interface.equality_scale` precisely so that ``penalty_term`` is the
dimensionless O(10) parameter Nitsche's theory calls for, in any unit system. The
tests below therefore ask for the correct flux at ``penalty_term = 1`` and for the
answer not to move across four decades of it. Before that scaling existed the mixed
Sievert/Henry case was out by a factor of 4 at ``penalty_term = 10`` and by a factor
of 280 at 1e3 -- non-monotonically, the adjoint term having been left unbalanced by a
stabilisation some eleven orders of magnitude too weak.

Both orderings of a mixed pair are covered, since :meth:`Interface.equality_scale`
has to pick the Henry side out of the pair rather than assume a position.
"""

from mpi4py import MPI

import numpy as np
import pytest
from scipy.optimize import brentq

import festim as F

T = 800.0  # K
L = 1e-3  # m, thickness of each slab
P_UP = 1e5  # Pa, pressure the upstream wall is in equilibrium with

# (D_0, E_D, K_0, E_K, solubility law). The metals follow Sieverts and their K is in
# H/m3/Pa^0.5; the molten salt follows Henry's law and its K is in H/m3/Pa
MATERIALS = {
    "tungsten": (4.1e-7, 0.39, 1.87e24, 1.04, "sievert"),
    "metal_2": (1.0e-7, 0.30, 5.0e23, 0.80, "sievert"),
    "salt": (9.3e-7, 0.42, 2.6e20, 0.0, "henry"),
}

# (upstream, downstream)
CASES = [
    ("tungsten", "metal_2"),  # matching laws
    ("tungsten", "salt"),  # Sievert -> Henry
    ("salt", "tungsten"),  # Henry -> Sievert
]


def properties(name):
    """(D, K, law) of a material at ``T``."""
    D_0, E_D, K_0, E_K, law = MATERIALS[name]
    return D_0 * np.exp(-E_D / (F.k_B * T)), K_0 * np.exp(-E_K / (F.k_B * T)), law


def partial_pressure(concentration, K, law):
    """The pressure a material at this concentration is in equilibrium with."""
    return (concentration / K) ** 2 if law == "sievert" else concentration / K


def wall_concentration(name):
    """Concentration of the upstream wall, in equilibrium with ``P_UP``."""
    _, K, law = properties(name)
    return K * np.sqrt(P_UP) if law == "sievert" else K * P_UP


def exact_flux(case):
    """Steady permeation flux with both sides in equilibrium at the interface.

    No source and no trapping, so each slab carries the same flux down a linear
    profile and the interface state follows from a scalar balance: the pressure the
    upstream side is left at, having given up ``flux * L / D``, must equal the one
    the downstream side reaches on receiving it.
    """
    upstream, downstream = case
    D_up, K_up, law_up = properties(upstream)
    D_down, K_down, law_down = properties(downstream)
    c_up = wall_concentration(upstream)

    def balance(flux):
        return partial_pressure(
            c_up - flux * L / D_up, K_up, law_up
        ) - partial_pressure(flux * L / D_down, K_down, law_down)

    # decreasing in flux, positive at 0, negative once the upstream side is emptied
    return brentq(balance, 0.0, c_up * D_up / L, xtol=1e-3, rtol=1e-14)


def build(method, penalty, case, n=40):
    """Two slabs of real material, fed at ``P_UP`` and empty downstream."""
    materials = []
    for name in case:
        D_0, E_D, K_0, E_K, law = MATERIALS[name]
        material = F.Material(D_0=D_0, E_D=E_D, K_S_0=K_0, E_K_S=E_K)
        material.solubility_law = law
        materials.append(material)

    upstream = F.VolumeSubdomain1D(id=1, borders=[0, L], material=materials[0])
    downstream = F.VolumeSubdomain1D(id=2, borders=[L, 2 * L], material=materials[1])
    left = F.SurfaceSubdomain1D(id=3, x=0)
    right = F.SurfaceSubdomain1D(id=4, x=2 * L)

    H = F.Species("H", subdomains=[upstream, downstream])
    vertices = np.unique(
        np.concatenate([np.linspace(0, L, n + 1), np.linspace(L, 2 * L, n + 1)])
    )
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh1D(vertices),
        species=[H],
        subdomains=[upstream, downstream, left, right],
        boundary_conditions=[
            F.FixedConcentrationBC(left, value=wall_concentration(case[0]), species=H),
            F.FixedConcentrationBC(right, value=0.0, species=H),
        ],
        interfaces=[
            F.Interface(5, (upstream, downstream), penalty_term=penalty, method=method)
        ],
        temperature=T,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )
    model.exports = [
        F.SurfaceFlux(field=H, surface=left),
        F.SurfaceFlux(field=H, surface=right),
    ]
    return model


def permeation_flux(method, penalty, case):
    """The flux in through the fed wall; also checks it equals the flux out.

    The conservation tolerance is relative to the *problem's* flux scale, not to the
    measured value: a penalty far too small for these materials leaves both walls at
    numerical noise, which no purely relative comparison can judge.
    """
    model = build(method, penalty, case)
    model.initialise()
    model.run()
    flux_in, flux_out = (export.data[-1] for export in model.exports)
    assert np.isclose(flux_in, -flux_out, rtol=1e-7, atol=1e-7 * exact_flux(case)), (
        "interface is not conservative"
    )
    return abs(flux_in)


def test_the_problem_is_posed_at_realistic_magnitudes():
    """Guard the point of the file: unit-scale data here would prove nothing."""
    assert 1e20 < wall_concentration("tungsten") < 1e21
    assert 1e25 < wall_concentration("salt") < 1e26
    for case in CASES:
        assert 1e11 < exact_flux(case) < 1e15
    # and the two solubility scales are decades and different units apart
    assert properties("tungsten")[1] < 1e18 < 1e20 < properties("salt")[1]


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
@pytest.mark.parametrize("penalty", [1.0, 10.0, 1e2, 1e4])
@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(c))
def test_nitsche_is_accurate_in_si_units_at_a_dimensionless_penalty(case, penalty):
    """Nitsche must not care what units the problem is written in.

    ``penalty_term`` is the stabilisation parameter here, not a conductance, so the
    same O(1) values that work at unit scale have to work at 1e25 H/m3 -- and the
    answer must be flat across four decades of it, a consistent method taking its
    accuracy from the consistency term rather than from the penalty.
    """
    assert np.isclose(
        permeation_flux("nitsche", penalty, case), exact_flux(case), rtol=1e-6
    )


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(c))
def test_penalty_term_is_a_conductance_and_must_beat_the_bulk(case):
    """The penalty's error is set by ``penalty_term * R``, R the bulk resistance.

    Documented rather than deplored: it is what a pure penalty does. The point is the
    magnitude -- ``1/R`` is of order 1e12 for these materials, so the default
    ``penalty_term = 10`` is ten orders of magnitude short and the interface blocks
    essentially everything.
    """
    D_up, K_up, law_up = properties(case[0])
    D_down, K_down, law_down = properties(case[1])
    R_total = L / (D_up * K_up) + L / (D_down * K_down)
    J_exact = exact_flux(case)

    assert permeation_flux("penalty", 10.0, case) < 1e-6 * J_exact

    # and it approaches the right answer only once penalty_term * R exceeds one
    errors = [
        abs(permeation_flux("penalty", scale / R_total, case) - J_exact) / J_exact
        for scale in (1.0, 1e2, 1e4)
    ]
    assert errors[0] > 1e-3
    assert errors[-1] < 1e-3
    assert np.all(np.diff(errors) < 0)

    if law_up == law_down:
        # matching laws couple linearly in c/K, so the series algebra is exact and
        # the relative error is 1/(1 + penalty_term * R) to four figures
        assert np.allclose(errors, [1 / 2, 1 / 101, 1 / 10001], rtol=1e-3)
