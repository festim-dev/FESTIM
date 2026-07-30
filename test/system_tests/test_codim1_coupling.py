"""Codimensional (manifold) coupling: a transport equation on a codim-1 subdomain of
the mesh, flux-coupled to the bulk. See issue #1208.

Manufactured solution, in the material coordinates ``xi`` (``xi = x`` in 2D, and the
inverse rotation of ``x`` in the tilted 3D case)::

    c_O(xi) = 1 + xi_0**2 + (1 + xi_0) cos(pi xi_1)
    c_G(xi) = 1 + beta cos(pi xi_1),        beta = 1 - D_O / k
    J       = k (c_O - c_G) = D_O cos(pi xi_1)

    Omega:  -D_O lap(c_O) = f_O,  with  -D_O grad(c_O).n = J on Gamma
    Gamma:  D_G lap_Gamma(c_G) + S + J = 0     (Omega loses J, Gamma gains it)

    f_O = -D_O (2 - pi**2 (1 + xi_0) cos(pi xi_1))
    S   = (D_G pi**2 beta - D_O) cos(pi xi_1)

It is chosen so every natural condition holds exactly: ``d c_O / d xi_1 = 0`` at
``xi_1 = 0, 1`` so Omega needs a single Dirichlet BC at ``xi_0 = 1``, and
``d c_G / d xi_1 = 0`` at the ends of Gamma, which matters because strong boundary
conditions on a manifold subdomain are not supported. ``J`` varies along Gamma, so the
coupling is genuinely exercised rather than constant.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F

from .tools import error_L2

D_O, D_G, K_EX = 1.5, 0.7, 2.0
BETA = 1.0 - D_O / K_EX
OMEGA_ID, GAMMA_ID, RIGHT_ID = 1, 2, 3


def rotation(alpha=0.6, gamma=0.4):
    """Rotation about z then about y, so that Gamma's normal is tilted with respect to
    every coordinate axis.

    This matters: under the parent facet measure ``ufl.grad`` of a manifold field
    happens to give the right answer for an axis-aligned Gamma, and only a tilted one
    exposes a wrong integration measure.
    """
    ca, sa, cg, sg = np.cos(alpha), np.sin(alpha), np.cos(gamma), np.sin(gamma)
    Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cg, 0.0, sg], [0.0, 1.0, 0.0], [-sg, 0.0, cg]])
    return Ry @ Rz


def run(n, dim=2, degree=1, velocity=None):
    """Solve the coupled problem and return the relative L2 errors on both fields."""
    if dim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
        R = np.eye(3)
    else:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
        R = rotation()
        mesh.geometry.x[:, :] = mesh.geometry.x @ R.T

    Rt = ufl.as_matrix(R.T[:dim, :dim].tolist())

    def xi(x):
        return Rt * ufl.as_vector([x[i] for i in range(dim)])

    # the ambient direction of increasing xi_1, tangent to Gamma
    t1 = (R @ np.array([0.0, 1.0, 0.0]))[:dim]
    v_np = np.zeros(dim) if velocity is None else np.asarray(velocity, dtype=float)
    v_dot_t1 = float(v_np @ t1)

    # Gamma is located in the *material* coordinates, ie. before the rotation
    def on_gamma(x):
        return np.isclose((R.T[:dim, :dim] @ x[:dim])[0], 0.0)

    def on_right(x):
        return np.isclose((R.T[:dim, :dim] @ x[:dim])[0], 1.0)

    omega = F.VolumeSubdomain(
        id=OMEGA_ID,
        material=F.Material(D_0=D_O, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    gamma = F.VolumeSubdomain(
        id=GAMMA_ID,
        material=F.Material(D_0=D_G, E_D=0.0),
        dim=dim - 1,
        locator=on_gamma,
    )
    right = F.SurfaceSubdomain(id=RIGHT_ID, locator=on_right)

    H_om = F.Species("H_om", subdomains=[omega])
    H_gam = F.Species("H_gam", subdomains=[gamma])

    pi = np.pi
    sources = [
        F.ParticleSource(
            value=lambda x: (
                -D_O * (2 - pi**2 * (1 + xi(x)[0]) * ufl.cos(pi * xi(x)[1]))
            ),
            species=H_om,
            volume=omega,
        ),
        F.ParticleSource(
            value=lambda x: (
                (D_G * pi**2 * BETA - D_O) * ufl.cos(pi * xi(x)[1])
                - pi * BETA * v_dot_t1 * ufl.sin(pi * xi(x)[1])
            ),
            species=H_gam,
            volume=gamma,
        ),
        # the manifold gains exactly what the bulk loses
        F.ParticleSource(
            value=lambda c_g, c_o: K_EX * (c_o - c_g),
            species=H_gam,
            volume=gamma,
            species_dependent_value={"c_o": H_om, "c_g": H_gam},
        ),
    ]
    bcs = [
        F.ParticleFluxBC(
            subdomain=gamma,
            value=lambda c_g, c_o: K_EX * (c_g - c_o),
            species=H_om,
            species_dependent_value={"c_o": H_om, "c_g": H_gam},
        ),
        F.FixedConcentrationBC(
            subdomain=right,
            value=lambda x: 1 + xi(x)[0] ** 2 + (1 + xi(x)[0]) * ufl.cos(pi * xi(x)[1]),
            species=H_om,
        ),
    ]

    advection = []
    if velocity is not None:
        vel = dolfinx.fem.Function(
            dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (dim,)))
        )
        vel.interpolate(lambda x: np.tile(v_np, (x.shape[1], 1)).T)
        advection = [F.AdvectionTerm(velocity=vel, subdomain=gamma, species=H_gam)]

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_om, H_gam],
        subdomains=[omega, gamma, right],
        sources=sources,
        boundary_conditions=bcs,
        advection_terms=advection,
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    model.initialise()
    model.run()

    def exact(sub):
        X = Rt * ufl.SpatialCoordinate(sub)
        return X

    c_om = H_om.subdomain_to_post_processing_solution[omega]
    c_gam = H_gam.subdomain_to_post_processing_solution[gamma]
    X_om = exact(omega.submesh)
    X_gam = exact(gamma.submesh)
    return (
        error_L2(c_om, 1 + X_om[0] ** 2 + (1 + X_om[0]) * ufl.cos(pi * X_om[1])),
        error_L2(c_gam, 1 + BETA * ufl.cos(pi * X_gam[1])),
    )


def _rates(errors, refinements):
    return [
        np.log(e0 / e1) / np.log(n1 / n0)
        for e0, e1, n0, n1 in zip(
            errors, errors[1:], refinements, refinements[1:], strict=False
        )
    ]


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_codim1_coupling_2d():
    """A 1D manifold in a 2D mesh converges at the expected rate."""
    refinements = [8, 16, 32]
    errors = [run(n, dim=2) for n in refinements]
    rates_om = _rates([e[0] for e in errors], refinements)
    rates_gam = _rates([e[1] for e in errors], refinements)

    assert all(r > 1.9 for r in rates_om), rates_om
    assert all(r > 1.9 for r in rates_gam), rates_gam


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_codim1_coupling_3d_tilted():
    """A 2D manifold in a 3D mesh, tilted with respect to every coordinate axis.

    This is the case that fails loudly (the manifold field stops converging) if the
    gradient terms of a codim-1 subdomain are ever integrated over the parent facet
    measure instead of a measure on the submesh.
    """
    refinements = [4, 8, 16]
    errors = [run(n, dim=3) for n in refinements]
    rates_om = _rates([e[0] for e in errors], refinements)
    rates_gam = _rates([e[1] for e in errors], refinements)

    assert all(r > 1.8 for r in rates_om), rates_om
    assert all(r > 1.8 for r in rates_gam), rates_gam


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_codim1_coupling_with_advection():
    """Advection along the manifold, on a tilted 3D Gamma."""
    refinements = [4, 8, 16]
    errors = [run(n, dim=3, velocity=(0.9, -0.5, 0.7)) for n in refinements]
    rates_gam = _rates([e[1] for e in errors], refinements)

    assert all(r > 1.8 for r in rates_gam), rates_gam


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_advection_velocity_is_self_projecting():
    """The normal component of an advection velocity on a manifold does nothing.

    The tangential gradient is orthogonal to the normal, so ``dot(grad(c), v)`` already
    picks out the tangential part of ``v`` and the user need not project it.
    """
    v = np.array([0.9, -0.5, 0.7])
    normal = rotation() @ np.array([1.0, 0.0, 0.0])
    projected = v - (v @ normal) * normal

    raw = run(8, dim=3, velocity=v)
    tangential = run(8, dim=3, velocity=projected)

    assert np.allclose(raw, tangential, rtol=1e-8, atol=0)
