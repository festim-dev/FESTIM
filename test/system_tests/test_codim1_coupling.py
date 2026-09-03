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
OMEGA_ID, GAMMA_ID, RIGHT_ID, GAMMA_ENDS_ID = 1, 2, 3, 4


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
    subdomains = [omega, gamma, right]
    if velocity is not None:
        vel = dolfinx.fem.Function(
            dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (dim,)))
        )
        vel.interpolate(lambda x: np.tile(v_np, (x.shape[1], 1)).T)
        advection = [F.AdvectionTerm(velocity=vel, subdomain=gamma, species=H_gam)]
        # the manufactured solution has zero *diffusive* flux on the whole of dGamma
        # (dc/dxi_2 vanishes at xi_2 = 0, 1 and c does not vary along xi_3), which is
        # the natural condition an outflow gives. Without it the divergence form would
        # impose zero *total* flux there, which the exact solution does not satisfy
        gamma_ends = F.SurfaceSubdomain(
            id=GAMMA_ENDS_ID,
            dim=dim - 2,
            locator=lambda x: np.full_like(x[0], True, dtype=bool),
        )
        subdomains.append(gamma_ends)
        bcs.append(F.OutflowBC(subdomain=gamma_ends, species=H_gam))

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_om, H_gam],
        subdomains=subdomains,
        sources=sources,
        boundary_conditions=bcs,
        drift_terms=advection,
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
def test_coupling_source_internal_subdomain():
    """Create a simple 2D codim-1 problem to verify codim-1 coupling terms are
    being applied correctly. Compares against a known solution.
    """
    D_BULK, D_GAMMA = 1.5, 0.7
    PLANE = 0.5

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 20, 20)

    left = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=D_BULK, E_D=0.0),
        locator=lambda x: x[0] <= PLANE + 1e-14,
    )
    right = F.VolumeSubdomain(
        id=2,
        material=F.Material(D_0=D_BULK, E_D=0.0),
        locator=lambda x: x[0] >= PLANE - 1e-14,
    )
    gamma = F.VolumeSubdomain(
        id=3,
        material=F.Material(D_0=D_GAMMA, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], PLANE),
    )
    outer_l = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 0.0))
    outer_r = F.SurfaceSubdomain(id=5, locator=lambda x: np.isclose(x[0], 1.0))

    H_l = F.Species("H_l", subdomains=[left])
    H_r = F.Species("H_r", subdomains=[right])
    H_g = F.Species("H_g", subdomains=[gamma])

    sources = [
        F.ParticleSource(
            value=lambda x, c_g, c_b: 4.0 * x[0] * (c_b - c_g),
            species=H_g,
            volume=gamma,
            species_dependent_value={"c_b": H_l, "c_g": H_g},
        ),
        F.ParticleSource(
            value=lambda x, c_g, c_b: 6.0 * x[0] * (c_b - c_g),
            species=H_g,
            volume=gamma,
            species_dependent_value={"c_b": H_r, "c_g": H_g},
        ),
    ]
    bcs = [
        F.ParticleFluxBC(
            subdomain=gamma,
            species=H_l,
            value=lambda x, c_g, c_b: 4.0 * x[0] * (c_g - c_b),
            species_dependent_value={"c_b": H_l, "c_g": H_g},
        ),
        F.ParticleFluxBC(
            subdomain=gamma,
            species=H_r,
            value=lambda x, c_g, c_b: 6.0 * x[0] * (c_g - c_b),
            species_dependent_value={"c_b": H_r, "c_g": H_g},
        ),
        F.FixedConcentrationBC(subdomain=outer_l, value=2.0, species=H_l),
        F.FixedConcentrationBC(subdomain=outer_r, value=0.0, species=H_r),
    ]

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_l, H_r, H_g],
        subdomains=[left, right, gamma, outer_l, outer_r],
        sources=sources,
        boundary_conditions=bcs,
        temperature=500,
        exports=[],
    )
    model.settings = F.Settings(atol=1e-12, rtol=1e-12, transient=False)
    model.initialise()
    model.run()

    c_l = H_l.subdomain_to_post_processing_solution[left].x.array
    c_r = H_r.subdomain_to_post_processing_solution[right].x.array
    c_g = H_g.subdomain_to_post_processing_solution[gamma].x.array

    assert np.allclose(c_l.min(), 14.0 / 9.0, rtol=1e-6, atol=1e-8)
    assert np.allclose(c_r.max(), 4.0 / 9.0, rtol=1e-6, atol=1e-8)
    assert np.allclose(c_g.mean(), 8.0 / 9.0, rtol=1e-6, atol=1e-8)


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


def uncoupled(stepsize, final_time, source_value, reactions=None, extra_species=()):
    """A manifold carrying its own equation and nothing else.

    Omega is inert (a single Dirichlet BC and no coupling to Gamma), which makes the
    manifold field a pure ODE with a closed-form backward-Euler solution. It also
    exercises the padding of Gamma's empty parent-mesh block.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)

    omega = F.VolumeSubdomain(
        id=OMEGA_ID,
        material=F.Material(D_0=D_O, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    gamma = F.VolumeSubdomain(
        id=GAMMA_ID,
        material=F.Material(D_0=D_G, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], 0.0),
    )
    right = F.SurfaceSubdomain(id=RIGHT_ID, locator=lambda x: np.isclose(x[0], 1.0))

    H_om = F.Species("H_om", subdomains=[omega])
    H_gam = F.Species("H_gam", subdomains=[gamma])

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_om, H_gam, *extra_species],
        subdomains=[omega, gamma, right],
        sources=[F.ParticleSource(value=source_value, species=H_gam, volume=gamma)],
        reactions=list(reactions or []),
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=right, value=0.0, species=H_om),
        ],
        temperature=500,
        settings=F.Settings(
            atol=1e-12,
            rtol=1e-12,
            transient=True,
            final_time=final_time,
            stepsize=stepsize,
        ),
    )
    return model, gamma, H_gam


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_transient_manifold_integrates_dt_exactly():
    """A transient manifold must use a timestep living on its own submesh.

    The self terms of a manifold are integrated over its submesh, and a parent-mesh
    ``fem.Constant`` there does not even compile (FFCx raises an undiagnosable
    ``UnboundLocalError``). Getting it to compile is not enough though: the mirror has
    to carry the current value, which a stepsize that keeps growing is what checks.

    With a spatially uniform source ``S`` and no coupling, the manifold field is
    ``c = S t`` and backward Euler integrates it exactly, so the solution reads back
    the sum of the timesteps actually used.
    """
    source_value = 3.0
    stepsize = F.Stepsize(
        initial_value=0.05,
        growth_factor=1.2,
        cutback_factor=0.8,
        target_nb_iterations=4,
    )
    model, gamma, H_gam = uncoupled(stepsize, 1.0, source_value)
    model.initialise()

    # the mirror is a constant of the submesh, not of the parent mesh
    assert gamma.sub_dt.ufl_domain() is gamma.submesh.ufl_domain()

    model.run()

    c_gam = H_gam.subdomain_to_post_processing_solution[gamma].x.array
    assert np.allclose(c_gam, source_value * float(model.t), rtol=1e-10)
    # the stepsize did grow, so a mirror stuck at the initial value would be caught
    assert float(model.dt) > 0.05


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_time_dependent_source_on_manifold():
    """An explicitly time-dependent source on a manifold needs a submesh-resident ``t``.

    Same trap as the timestep: the source expression is built once and integrated over
    the submesh, so the time it closes over cannot be the parent-mesh constant.

    ``S = t (1 + x)`` is written in terms of ``x`` deliberately -- naming ``x`` is what
    sends FESTIM down the mapped-expression path, where the time constant is baked into
    the form rather than re-interpolated. Gamma is the plane ``x = 0``, so the source is
    uniform along it and there is nothing for diffusion to flatten: backward Euler then
    gives exactly ``c = dt sum(t_n)``.
    """
    dt = 0.1
    model, gamma, H_gam = uncoupled(
        F.Stepsize(initial_value=dt),
        0.5,
        lambda x, t: t * (1 + x[0]),
    )
    model.initialise()
    assert gamma.sub_t.ufl_domain() is gamma.submesh.ufl_domain()
    model.run()

    n_steps = round(float(model.t) / dt)
    expected = dt**2 * n_steps * (n_steps + 1) / 2

    c_gam = H_gam.subdomain_to_post_processing_solution[gamma].x.array
    assert np.allclose(c_gam, expected, rtol=1e-10)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_reaction_on_manifold():
    """A reaction on a manifold reads the temperature inside a submesh integral.

    The parent-mesh temperature is the third coefficient that cannot appear there, so
    a manifold carrying a trap is the case that exercises it.

    A uniform source ``S`` feeding a first-order trapping reaction gives
    ``c_t' = k (S t - c_t) - p c_t``, whose transient decays like ``exp(-(k + p) t)``
    and leaves ``c_t = k S / (k + p) (t - 1 / (k + p))``. That asymptote is linear in
    ``t``, which backward Euler integrates exactly, so it can be asserted tightly.

    The activation energies are non-zero on purpose: with ``E = 0`` the Arrhenius factor
    folds to 1 and the temperature drops out of the form altogether, which would leave
    the coefficient under test unexercised.
    """
    temperature, source_value, final_time = 500.0, 1.0, 4.0
    k, p, E_k, E_p = 2.0, 5.0, 0.2, 0.3
    # the pre-exponential factors that put the Arrhenius rates at k and p
    k_0 = k * np.exp(E_k / (F.k_B * temperature))
    p_0 = p * np.exp(E_p / (F.k_B * temperature))

    trapped = F.Species("trapped", mobile=False)
    model, gamma, H_gam = uncoupled(
        F.Stepsize(initial_value=0.05),
        final_time,
        source_value,
        extra_species=[trapped],
    )
    model.temperature = temperature
    trapped.subdomains = [gamma]
    model.reactions = [
        F.Reaction(
            reactant=H_gam,
            product=trapped,
            k_0=k_0,
            E_k=E_k,
            p_0=p_0,
            E_p=E_p,
            volume=gamma,
        )
    ]
    model.initialise()
    model.run()

    c_mobile = H_gam.subdomain_to_post_processing_solution[gamma].x.array
    c_trapped = trapped.subdomain_to_post_processing_solution[gamma].x.array

    t = float(model.t)
    # everything the source put in is still on the manifold, split between the two
    assert np.allclose(c_mobile + c_trapped, source_value * t, rtol=1e-8)
    # and the reaction has split it in the proportion its rates dictate
    expected_trapped = k * source_value / (k + p) * (t - 1 / (k + p))
    assert np.allclose(c_trapped, expected_trapped, rtol=1e-6)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_implicit_species_on_manifold():
    """The density of an implicit species must live on the manifold's submesh.

    It is the fourth coefficient of a manifold self term, after ``dt``, ``t`` and the
    temperature, and it is reached by the usual way of writing trapping: a reaction
    consuming ``F.ImplicitSpecies`` empty sites. Built on the parent mesh it does not
    compile (the same FFCx ``UnboundLocalError``).

    Compiling is not the whole test though -- the density has to carry the right value
    too. With a uniform source ``S`` and saturable trapping, the two fields obey::

        c_m' = S - R,   c_t' = R,   R = k c_m (n - c_t) - p c_t

    Summing them removes ``R``, so ``c_m + c_t = S t`` exactly under backward Euler,
    and substituting ``c_m = S t - c_t`` leaves a scalar quadratic per step. Solving
    that quadratic here reproduces what the solver must return to machine precision,
    and ``n`` enters its coefficients directly, so a stale or wrong density shows up.
    """
    temperature, source_value, final_time, dt = 500.0, 1.0, 2.0, 0.1
    n_trap, k, p, E_k, E_p = 0.5, 2.0, 5.0, 0.2, 0.3
    k_0 = k * np.exp(E_k / (F.k_B * temperature))
    p_0 = p * np.exp(E_p / (F.k_B * temperature))

    trapped = F.Species("trapped", mobile=False)
    empty_sites = F.ImplicitSpecies(n=n_trap, others=[trapped], name="empty_sites")
    model, gamma, H_gam = uncoupled(
        F.Stepsize(initial_value=dt),
        final_time,
        source_value,
        extra_species=[trapped],
    )
    model.temperature = temperature
    trapped.subdomains = [gamma]
    model.reactions = [
        F.Reaction(
            reactant=[H_gam, empty_sites],
            product=trapped,
            k_0=k_0,
            E_k=E_k,
            p_0=p_0,
            E_p=E_p,
            volume=gamma,
        )
    ]
    model.initialise()

    # the density is a coefficient of the submesh integral, so it must live there
    assert empty_sites.value_fenics.ufl_domain() is gamma.submesh.ufl_domain()

    model.run()

    c_mobile = H_gam.subdomain_to_post_processing_solution[gamma].x.array
    c_trapped = trapped.subdomain_to_post_processing_solution[gamma].x.array

    # step the reference quadratic  a y**2 - (a b + a n + p dt + 1) y + (a b n + y_n)
    # with a = k dt and b = S t, taking the root below n
    expected_trapped, a = 0.0, k * dt
    for step in range(round(final_time / dt)):
        b = source_value * (step + 1) * dt
        beta = a * b + a * n_trap + p * dt + 1
        expected_trapped = (
            beta - np.sqrt(beta**2 - 4 * a * (a * b * n_trap + expected_trapped))
        ) / (2 * a)

    t = float(model.t)
    assert np.allclose(c_mobile + c_trapped, source_value * t, rtol=1e-8)
    assert np.allclose(c_trapped, expected_trapped, rtol=1e-8)
    # a good fraction of the sites are occupied, so the density is load-bearing:
    # the saturation term k c_m (n - c_t) is nowhere near its unsaturated limit
    assert 0.3 * n_trap < expected_trapped < 0.9 * n_trap


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_implicit_species_shared_across_meshes_raises():
    """One implicit species cannot serve reactions integrated over different meshes.

    Its density is a single fenics object built on one mesh; used by a reaction on a
    manifold and by one on a bulk subdomain, it would silently be rebuilt on whichever
    mesh came last and leave a foreign terminal in the other integral.
    """
    trapped_om = F.Species("trapped_om", mobile=False)
    trapped_gam = F.Species("trapped_gam", mobile=False)
    shared = F.ImplicitSpecies(n=0.5, others=[trapped_om, trapped_gam], name="shared")
    model, gamma, H_gam = uncoupled(
        F.Stepsize(initial_value=0.1),
        0.1,
        1.0,
        extra_species=[trapped_om, trapped_gam],
    )
    omega = next(s for s in model.volume_subdomains if s.id == OMEGA_ID)
    H_om = next(s for s in model.species if s.name == "H_om")
    trapped_om.subdomains = [omega]
    trapped_gam.subdomains = [gamma]
    model.reactions = [
        F.Reaction(
            reactant=[H_gam, shared],
            product=trapped_gam,
            k_0=1.0,
            E_k=0.0,
            p_0=1.0,
            E_p=0.0,
            volume=gamma,
        ),
        F.Reaction(
            reactant=[H_om, shared],
            product=trapped_om,
            k_0=1.0,
            E_k=0.0,
            p_0=1.0,
            E_p=0.0,
            volume=omega,
        ),
    ]

    with pytest.raises(
        NotImplementedError, match="a separate implicit species for each subdomain"
    ):
        model.initialise()


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_implicit_species_density_on_the_wrong_mesh_raises():
    """A density given as a ready-made fenics object cannot be moved to the submesh.

    Floats and callables are built on whatever mesh FESTIM asks for, but a
    ``fem.Function`` is passed through as it is, so one defined on the parent mesh
    reaches the submesh integral and dies inside FFCx. That has to be said plainly.
    """
    trapped = F.Species("trapped", mobile=False)
    model, gamma, H_gam = uncoupled(
        F.Stepsize(initial_value=0.1), 0.1, 1.0, extra_species=[trapped]
    )
    V = dolfinx.fem.functionspace(model.mesh.mesh, ("Lagrange", 1))
    parent_density = dolfinx.fem.Function(V)
    parent_density.x.array[:] = 0.5

    trapped.subdomains = [gamma]
    model.reactions = [
        F.Reaction(
            reactant=[H_gam, F.ImplicitSpecies(n=parent_density, others=[trapped])],
            product=trapped,
            k_0=1.0,
            E_k=0.0,
            p_0=1.0,
            E_p=0.0,
            volume=gamma,
        )
    ]

    with pytest.raises(NotImplementedError, match="defined on another mesh"):
        model.initialise()


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
