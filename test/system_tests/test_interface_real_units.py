"""Interface coupling posed in SI units, with real material data.

Every other interface test in this directory is written at unit scale -- D, K_S and
concentrations all of order one -- which hides anything that depends on the *units*
of the coupling rather than on its structure. Real hydrogen transport is nothing like
that: at 600 K the numbers below span ``D ~ 1e-10 m2/s``, ``K_S ~ 1e15-1e17``,
``c ~ 1e18 H/m3`` and a permeation flux ``~1e11 H/m2/s``.

Two properties are pinned here.

``penalty_term`` is a **surface conductance** for :meth:`Interface.penalty_method`:
the interface imposes ``flux = penalty_term * equality``, so it acts as a contact
resistance ``1/penalty_term`` in series with the bulk ``R = L/(D K)``, and the only
thing that matters is the dimensionless product ``penalty_term * R``. With the data
below ``1/R ~ 7e8``, so the default ``penalty_term = 10`` transmits about a
billionth of the correct flux -- a perfect barrier, silently. That is a property of
a pure penalty, not a bug, but it has to be stated somewhere a reader will find it.

:meth:`Interface.nitsche_method` must have no such dependence. Its stabilisation is
scaled by :meth:`Interface.equality_scale` precisely so that ``penalty_term`` is the
dimensionless O(10) parameter Nitsche's theory calls for, in any unit system. The
test below therefore asks for the correct flux at ``penalty_term = 10`` and for the
answer not to move across four decades of it. Before that scaling existed, the
mixed Sievert/Henry case was out by a factor of 4 at ``penalty_term = 10`` and by a
factor of 280 at 1e3 -- non-monotonically, since the adjoint term was left
unbalanced by a stabilisation that was ~1e11 too weak.
"""

from mpi4py import MPI

import numpy as np
import pytest

import festim as F

T = 600.0  # K
L = 1e-3  # m, thickness of each slab
P_UP = 1e5  # Pa, upstream pressure

# tungsten (Frauenfelder) and a second metal with a higher solubility
D_0_W, E_D_W, K_S_0_W, E_K_S_W = 4.1e-7, 0.39, 1.87e24, 1.04
D_0_X, E_D_X, K_S_0_X, E_K_S_X = 1.0e-7, 0.30, 5.0e23, 0.80

D_W = D_0_W * np.exp(-E_D_W / (F.k_B * T))
K_W = K_S_0_W * np.exp(-E_K_S_W / (F.k_B * T))
D_X = D_0_X * np.exp(-E_D_X / (F.k_B * T))
K_X = K_S_0_X * np.exp(-E_K_S_X / (F.k_B * T))
C_UP = K_W * np.sqrt(P_UP)  # Sievert equilibrium at the upstream wall


def exact_flux(laws):
    """Steady permeation flux with the two sides in equilibrium at the interface.

    Both slabs carry the same flux and the downstream wall is empty, so the
    interface state follows from a scalar balance. With matching laws the coupling
    is linear in ``c/K`` and the slabs are two resistances in series; with a
    Sievert upstream and a Henry downstream the balance is quadratic in
    ``sqrt(P_interface)``.
    """
    if laws[0] == laws[1]:
        return (C_UP / K_W) / (L / (D_W * K_W) + L / (D_X * K_X))

    # D_W K_W (sqrt(P_up) - y) = D_X K_X y**2, with y = sqrt(P_interface)
    a, b = D_X * K_X, D_W * K_W
    y = (-b + np.sqrt(b**2 + 4 * a * b * np.sqrt(P_UP))) / (2 * a)
    return a * y**2 / L


def build(method, penalty, laws, n=40):
    """Two slabs of real material, fed at ``P_UP`` and empty downstream."""
    material_up = F.Material(D_0=D_0_W, E_D=E_D_W, K_S_0=K_S_0_W, E_K_S=E_K_S_W)
    material_down = F.Material(D_0=D_0_X, E_D=E_D_X, K_S_0=K_S_0_X, E_K_S=E_K_S_X)
    material_up.solubility_law, material_down.solubility_law = laws

    upstream = F.VolumeSubdomain1D(id=1, borders=[0, L], material=material_up)
    downstream = F.VolumeSubdomain1D(id=2, borders=[L, 2 * L], material=material_down)
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
            F.FixedConcentrationBC(left, value=C_UP, species=H),
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


def permeation_flux(method, penalty, laws):
    """The flux in through the fed wall; also checks it equals the flux out.

    The tolerance is relative to the *problem's* flux scale, not to the measured
    value: a penalty far too small for these materials leaves both walls at
    numerical noise, which no purely relative comparison can judge.
    """
    model = build(method, penalty, laws)
    model.initialise()
    model.run()
    flux_in, flux_out = (export.data[-1] for export in model.exports)
    assert np.isclose(flux_in, -flux_out, rtol=1e-8, atol=1e-8 * exact_flux(laws)), (
        "interface is not conservative"
    )
    return abs(flux_in)


LAW_PAIRS = [("sievert", "sievert"), ("sievert", "henry")]


def test_reference_fluxes():
    """Pin the closed forms themselves before using them as references."""
    assert np.isclose(exact_flux(("sievert", "sievert")), 2.301616e11, rtol=1e-6)
    assert np.isclose(exact_flux(("sievert", "henry")), 2.340009e11, rtol=1e-6)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
@pytest.mark.parametrize("penalty", [1.0, 10.0, 1e2, 1e4])
@pytest.mark.parametrize("laws", LAW_PAIRS, ids=lambda p: "-".join(p))
def test_nitsche_is_accurate_in_si_units_at_a_dimensionless_penalty(laws, penalty):
    """Nitsche must not care what units the problem is written in.

    ``penalty_term`` here is the stabilisation parameter, not a conductance, so the
    same O(10) values that work at unit scale have to work at 1e18 H/m3 -- and the
    answer must be flat across four decades of it, since a consistent method takes
    its accuracy from the consistency term rather than from the penalty.
    """
    assert np.isclose(
        permeation_flux("nitsche", penalty, laws), exact_flux(laws), rtol=1e-3
    )


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
@pytest.mark.parametrize("laws", LAW_PAIRS, ids=lambda p: "-".join(p))
def test_penalty_term_is_a_conductance_and_must_beat_the_bulk(laws):
    """The penalty's error is ``1/(1 + penalty_term * R)``, R the bulk resistance.

    Documented rather than deplored: it is what a pure penalty does. The point is
    the magnitude -- ``1/R`` is about 7e8 for these materials, so the default
    ``penalty_term = 10`` is nine orders of magnitude short and the interface blocks
    essentially everything.
    """
    R_total = L / (D_W * K_W) + L / (D_X * K_X)
    J_exact = exact_flux(laws)

    assert permeation_flux("penalty", 10.0, laws) < 1e-6 * J_exact

    # and it approaches the right answer only once penalty_term * R exceeds one
    errors = []
    for penalty in (1 / R_total, 1e2 / R_total, 1e4 / R_total):
        errors.append(
            abs(permeation_flux("penalty", penalty, laws) - J_exact) / J_exact
        )
    assert errors[-1] < 1e-3
    assert np.all(np.diff(errors) < 0)

    if laws[0] == laws[1]:
        # matching laws couple linearly in c/K, so the series algebra is exact and
        # the relative error is 1/(1 + penalty_term * R) to four figures
        assert np.allclose(errors, [1 / 2, 1 / 101, 1 / 10001], rtol=1e-3)
    else:
        # a mixed pair couples through pressure instead, so R above is no longer the
        # matching resistance -- the same decay, on a shifted scale
        assert np.allclose(errors, [0.0474, 0.0016, 0.0], atol=1e-4)
