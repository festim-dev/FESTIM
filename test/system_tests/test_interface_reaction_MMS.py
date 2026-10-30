"""MMS convergence tests for festim.InterfaceReaction.

Each exact solution is built so that the interface conditions hold exactly: the
product profile ``q`` is chosen freely, the rate ``R`` follows from the product flux
condition ``-D_1 q'(a) = R``, and the reactant profile ``p`` is then given the value
and slope at the interface that satisfy the rate law and the reactant flux condition
``-D_0 p'(a) = n R``, with ``n`` the stoichiometry of the reactant.
"""

import math

import numpy as np
import pytest
import ufl

import festim as F

from .test_multi_material import generate_mesh
from .tools import error_L2

D_0, D_1 = 2.0, 5.0
K_PLUS, K_MINUS = 3.0, 4.0
A = 0.5  # position of the interface


def observed_order(sizes, errors):
    """Slope of log(error) against log(h)."""
    return np.polyfit(np.log(1 / np.array(sizes)), np.log(errors), 1)[0]


def reactant_profile(value, slope, mod):
    """A non-polynomial profile taking ``value`` and ``slope`` at the interface."""
    return lambda s: value + slope * (s - A) + (s - A) ** 2 * (1 + mod.sin(s))


def solve_first_order_2d(n):
    """2D steady first-order exchange A <=> B across y = 0.5, R = k+ A - k- B.

    Returns the L2 errors of A (bottom) and B (top)."""
    q = lambda y, mod: 1 + y + mod.sin(2 * mod.pi * y)  # noqa: E731
    rate = -D_1 * (1 - 2 * math.pi)  # -D_1 q'(A)
    p_value = (rate + K_MINUS * q(A, math)) / K_PLUS
    p_slope = -rate / D_0

    def exact(mod, bottom):
        def u(x):
            g = 2 + mod.cos(mod.pi * x[0])  # zero flux on x = 0 and x = 1
            profile = reactant_profile(p_value, p_slope, mod) if bottom else None
            return g * (profile(x[1]) if bottom else q(x[1], mod))

        return u

    mesh, mt, ct = generate_mesh(n)
    bottom = F.VolumeSubdomain(3, material=F.Material(D_0=D_0, E_D=0))
    top = F.VolumeSubdomain(4, material=F.Material(D_0=D_1, E_D=0))
    top_surface, bottom_surface = F.SurfaceSubdomain(id=1), F.SurfaceSubdomain(id=2)
    species_A = F.Species("A", subdomains=[bottom])
    species_B = F.Species("B", subdomains=[top])

    def source(u, D):
        return lambda x: -ufl.div(D * ufl.grad(u(x)))

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        subdomains=[bottom, top, top_surface, bottom_surface],
        species=[species_A, species_B],
        interfaces=[
            F.InterfaceReaction(
                id=5,
                subdomains=[bottom, top],
                k_plus=K_PLUS,
                k_minus=K_MINUS,
                reactants=[species_A],
                products=[species_B],
            )
        ],
        boundary_conditions=[
            F.FixedConcentrationBC(bottom_surface, exact(ufl, True), species_A),
            F.FixedConcentrationBC(top_surface, exact(ufl, False), species_B),
        ],
        sources=[
            F.ParticleSource(source(exact(ufl, True), D_0), bottom, species_A),
            F.ParticleSource(source(exact(ufl, False), D_1), top, species_B),
        ],
        temperature=300,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )
    model.volume_meshtags = ct
    model.facet_meshtags = mt
    model.initialise()
    model.run()

    return (
        error_L2(
            species_A.subdomain_to_post_processing_solution[bottom], exact(np, True)
        ),
        error_L2(
            species_B.subdomain_to_post_processing_solution[top], exact(np, False)
        ),
    )


def solve_recombination_1d(n):
    """1D steady recombination A + A <=> B across x = 0.5, R = k+ A^2 - k- B, with
    A consumed at 2 R.

    Returns the L2 errors of A (left) and B (right)."""
    q = lambda x, mod: 1 + x + 0.5 * mod.sin(2 * mod.pi * x)  # noqa: E731
    rate = -D_1 * (1 - math.pi)  # -D_1 q'(A)
    p_value = math.sqrt((rate + K_MINUS * q(A, math)) / K_PLUS)
    p_slope = -2 * rate / D_0

    def exact_A(mod):
        return lambda x: reactant_profile(p_value, p_slope, mod)(x[0])

    def exact_B(mod):
        return lambda x: q(x[0], mod)

    left = F.VolumeSubdomain1D(
        id=1, material=F.Material(D_0=D_0, E_D=0), borders=[0, A]
    )
    right = F.VolumeSubdomain1D(
        id=2, material=F.Material(D_0=D_1, E_D=0), borders=[A, 1]
    )
    left_surface = F.SurfaceSubdomain1D(id=4, x=0)
    right_surface = F.SurfaceSubdomain1D(id=5, x=1)
    species_A = F.Species("A", subdomains=[left])
    species_B = F.Species("B", subdomains=[right])

    def source(u, D):
        return lambda x: -ufl.div(D * ufl.grad(u(x)))

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh1D(np.linspace(0, 1, n + 1)),
        subdomains=[left, right, left_surface, right_surface],
        species=[species_A, species_B],
        interfaces=[
            F.InterfaceReaction(
                id=3,
                subdomains=[left, right],
                k_plus=K_PLUS,
                k_minus=K_MINUS,
                reactants=[species_A, species_A],
                products=[species_B],
            )
        ],
        boundary_conditions=[
            F.FixedConcentrationBC(left_surface, exact_A(ufl), species_A),
            F.FixedConcentrationBC(right_surface, exact_B(ufl), species_B),
        ],
        sources=[
            F.ParticleSource(source(exact_A(ufl), D_0), left, species_A),
            F.ParticleSource(source(exact_B(ufl), D_1), right, species_B),
        ],
        temperature=300,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    model.initialise()
    model.run()

    return (
        error_L2(species_A.subdomain_to_post_processing_solution[left], exact_A(np)),
        error_L2(species_B.subdomain_to_post_processing_solution[right], exact_B(np)),
    )


@pytest.mark.parametrize(
    "solve, sizes",
    [(solve_first_order_2d, [8, 16, 32]), (solve_recombination_1d, [8, 16, 32, 64])],
    ids=["first-order-2d", "recombination-1d"],
)
def test_interface_reaction_MMS_converges_at_second_order(solve, sizes):
    """Test that the L2 error of both species converges at second order (P1)."""
    errors = np.array([solve(n) for n in sizes])
    orders = [observed_order(sizes, errors[:, side]) for side in (0, 1)]
    assert min(orders) > 1.9
