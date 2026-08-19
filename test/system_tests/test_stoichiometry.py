import numpy as np
import ufl
from dolfinx import fem

import festim as F

from .tools import error_L2, test_mesh_1d, x_1d

# reaction rate constants chosen so that the reaction term is of the same order
# as the diffusion term: a wrong stoichiometric coefficient then shows up as a
# large error rather than a small perturbation
k_0 = 50
E_k = 0.1
p_0 = 5
E_p = 0.05

D_0 = 2
E_D = 0.1


def T_expr(x):
    return 500 + 100 * x[0]


def rate_coefficients(mesh):
    """The diffusion coefficient and the forward/backward rate coefficients as ufl
    expressions, built on the same temperature field as the model."""
    V = fem.functionspace(mesh, ("Lagrange", 1))
    T = fem.Function(V)
    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))
    k = k_0 * ufl.exp(-E_k / (F.k_B * T))
    p = p_0 * ufl.exp(-E_p / (F.k_B * T))
    return D, k, p


def test_repeated_reactant_MMS_steady_state():
    """MMS test for a reaction consuming two of the same reactant, 2A <--> B.

    A is consumed at rate 2R and B produced at rate R, with R = k * c_A**2 - p * c_B.
    The manufactured source for A carries the factor 2, so a stoichiometric
    coefficient of -1 instead of -2 for the repeated reactant leaves the exact
    solution far from the computed one.
    """

    def u_exact(mod):
        return lambda x: 1.5 + mod.sin(3 * mod.pi * x[0])

    def v_exact(mod):
        return lambda x: 2 + mod.cos(2 * mod.pi * x[0])

    A_analytical_ufl = u_exact(ufl)
    A_analytical_np = u_exact(np)
    B_analytical_ufl = v_exact(ufl)
    B_analytical_np = v_exact(np)

    D, k, p = rate_coefficients(test_mesh_1d.mesh)

    # net reaction rate of the manufactured solution
    R = k * A_analytical_ufl(x_1d) ** 2 - p * B_analytical_ufl(x_1d)

    # A is mobile and consumed at rate 2R, B is immobile and produced at rate R
    f = -ufl.div(D * ufl.grad(A_analytical_ufl(x_1d))) + 2 * R
    g = -R

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=2, x=0)
    right = F.SurfaceSubdomain1D(id=3, x=1)
    my_model.subdomains = [vol, left, right]

    A = F.Species("A")
    B = F.Species("B", mobile=False)
    my_model.species = [A, B]

    my_model.reactions = [
        F.ArrheniusReaction(
            reactant=[A, A],
            product=B,
            k_0=k_0,
            E_k=E_k,
            p_0=p_0,
            E_p=E_p,
            volume=vol,
        )
    ]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=A_analytical_ufl, species=A),
        F.DirichletBC(subdomain=right, value=A_analytical_ufl, species=A),
    ]

    my_model.sources = [
        F.ParticleSource(value=f, volume=vol, species=A),
        F.ParticleSource(value=g, volume=vol, species=B),
    ]

    my_model.settings = F.Settings(atol=1e-12, rtol=1e-12, transient=False)

    my_model.initialise()
    my_model.run()

    L2_error_A = error_L2(A.post_processing_solution, A_analytical_np)
    L2_error_B = error_L2(B.post_processing_solution, B_analytical_np)

    assert L2_error_A < 1e-5
    assert L2_error_B < 1e-5


def test_repeated_product_MMS_steady_state():
    """MMS test for a reaction producing two of the same product, A <--> 2B.

    A is consumed at rate R and B produced at rate 2R, with R = k * c_A - p * c_B**2.
    The manufactured source for B carries the factor 2, so a stoichiometric
    coefficient of +1 instead of +2 for the repeated product leaves the exact
    solution far from the computed one.
    """

    def u_exact(mod):
        return lambda x: 1.5 + mod.sin(3 * mod.pi * x[0])

    def v_exact(mod):
        return lambda x: 2 + mod.cos(2 * mod.pi * x[0])

    A_analytical_ufl = u_exact(ufl)
    A_analytical_np = u_exact(np)
    B_analytical_ufl = v_exact(ufl)
    B_analytical_np = v_exact(np)

    D, k, p = rate_coefficients(test_mesh_1d.mesh)

    # net reaction rate of the manufactured solution
    R = k * A_analytical_ufl(x_1d) - p * B_analytical_ufl(x_1d) ** 2

    # both species are mobile (B is kept mobile so that the diffusion term keeps the
    # Jacobian of the rate, quadratic in c_B, non-singular at the zero initial guess)
    # A is consumed at rate R and B produced at rate 2R
    f = -ufl.div(D * ufl.grad(A_analytical_ufl(x_1d))) + R
    g = -ufl.div(D * ufl.grad(B_analytical_ufl(x_1d))) - 2 * R

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=2, x=0)
    right = F.SurfaceSubdomain1D(id=3, x=1)
    my_model.subdomains = [vol, left, right]

    A = F.Species("A")
    B = F.Species("B")
    my_model.species = [A, B]

    my_model.reactions = [
        F.ArrheniusReaction(
            reactant=A,
            product=[B, B],
            k_0=k_0,
            E_k=E_k,
            p_0=p_0,
            E_p=E_p,
            volume=vol,
        )
    ]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=A_analytical_ufl, species=A),
        F.DirichletBC(subdomain=right, value=A_analytical_ufl, species=A),
        F.DirichletBC(subdomain=left, value=B_analytical_ufl, species=B),
        F.DirichletBC(subdomain=right, value=B_analytical_ufl, species=B),
    ]

    my_model.sources = [
        F.ParticleSource(value=f, volume=vol, species=A),
        F.ParticleSource(value=g, volume=vol, species=B),
    ]

    my_model.settings = F.Settings(atol=1e-12, rtol=1e-12, transient=False)

    my_model.initialise()
    my_model.run()

    L2_error_A = error_L2(A.post_processing_solution, A_analytical_np)
    L2_error_B = error_L2(B.post_processing_solution, B_analytical_np)

    assert L2_error_A < 1e-5
    assert L2_error_B < 1e-5
