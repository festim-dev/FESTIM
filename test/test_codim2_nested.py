"""Nested codimensional coupling: a transport equation on a codim-2 subdomain, which is
codim-1 *relative to the manifold it lies in*, exchanging with that manifold.

The nesting is what keeps the problem well posed. Coupling a line in a 3D mesh directly
to the bulk would need the trace of an H1 bulk field on a set of codimension 2, which
does not exist (such sets have zero H1-capacity), and dually a Dirac measure on a line
is not in H^-1: the solution would have a log singularity and would not lie in the space
the problem is posed in. Nesting replaces that by two ordinary codim-1 traces
(bulk -> Gamma, then Gamma -> Lambda), each of which is the standard
H1 -> H1/2 trace, so no weighted spaces or averaging radius are needed.

Geometry, in all tests below::

    Omega   = unit cube                     (codim 0)
    Gamma   = the plane z = 1/2             (codim 1, area 1)
    Lambda  = the line y = z = 1/2          (codim 2, length 1, interior to Gamma)
"""

from itertools import pairwise

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F

OMEGA_ID, GAMMA_ID, LAMBDA_ID, OUTER_ID, END_ID = 1, 2, 3, 4, 5


def cube(n):
    """A unit cube whose facets contain the planes z = 1/2 and y = 1/2."""
    if n % 2:
        raise ValueError("n must be even for z = 1/2 to be a plane of the mesh")
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)


def geometry(mesh, D_omega=1.0, D_gamma=1.0, D_lambda=1.0):
    """Omega, Gamma and the nested Lambda, with no species attached yet."""
    omega = F.VolumeSubdomain(
        id=OMEGA_ID,
        material=F.Material(D_0=D_omega, E_D=0.0),
        locator=lambda x: np.full(x.shape[1], True),
        name="omega",
    )
    gamma = F.VolumeSubdomain(
        id=GAMMA_ID,
        material=F.Material(D_0=D_gamma, E_D=0.0),
        dim=2,
        locator=lambda x: np.isclose(x[2], 0.5),
        name="gamma",
    )
    # codim 2: dim = mesh_dim - 2, and a parent it is nested in. Its locator is applied
    # to *gamma's submesh*, not to the parent mesh
    lam = F.VolumeSubdomain(
        id=LAMBDA_ID,
        material=F.Material(D_0=D_lambda, E_D=0.0),
        dim=1,
        parent=gamma,
        locator=lambda x: np.isclose(x[1], 0.5),
        name="lambda",
    )
    return omega, gamma, lam


def integrate(field, subdomain):
    """The integral of a species over a subdomain's own mesh."""
    c = field.subdomain_to_post_processing_solution[subdomain]
    mesh = subdomain.submesh
    local = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(c * ufl.Measure("dx", domain=mesh))
    )
    return mesh.comm.allreduce(local, op=MPI.SUM)


# ---------------------------------------------------------------------------
# 1. the exchange itself
# ---------------------------------------------------------------------------


def run_exchange(n=8, k=5.0, final_time=4.0, dt=0.05):
    """Gamma starts full, Lambda empty, and the two exchange with rate ``k``.

    Nothing else acts: no source, no boundary condition, no exchange with the bulk.
    The total amount on Gamma plus the total amount on Lambda is therefore
    conserved, and at equilibrium both fields are uniform and equal, so their
    common value is ``area(Gamma) / (area(Gamma) + length(Lambda)) = 1/2``.

    This is a sharp test of the coupling: the sign convention, the measure the exchange
    is integrated over (Gamma's submesh, not the parent mesh), the entity map relating
    Lambda's submesh to Gamma's, and the extra assembly group. Getting the measure wrong
    breaks conservation; getting the sign wrong breaks the equilibrium value.
    """
    mesh = cube(n)
    omega, gamma, lam = geometry(mesh)

    # Omega needs a species of its own -- a subdomain with no species has no function
    # space -- but it is untouched here and stays at zero
    H_omega = F.Species("c_omega", subdomains=[omega])
    H_gamma = F.Species("c_gamma", subdomains=[gamma])
    H_lambda = F.Species("c_lambda", subdomains=[lam])

    def exchange(c_gamma, c_lambda):
        """The rate at which Lambda gains particles from Gamma, per unit length."""
        return k * (c_gamma - c_lambda)

    deps = {"c_gamma": H_gamma, "c_lambda": H_lambda}
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_omega, H_gamma, H_lambda],
        subdomains=[omega, gamma, lam],
        initial_conditions=[
            F.InitialConcentration(value=1.0, volume=gamma, species=H_gamma),
        ],
        sources=[
            # Lambda gains
            F.ParticleSource(
                value=exchange,
                species=H_lambda,
                volume=lam,
                species_dependent_value=deps,
            ),
        ],
        boundary_conditions=[
            # ... and Gamma loses exactly the same amount. A codim-2 subdomain is used
            # wherever a surface is expected, as a codim-1 one already is
            F.ParticleFluxBC(
                subdomain=lam,
                species=H_gamma,
                value=lambda c_gamma, c_lambda: -exchange(c_gamma, c_lambda),
                species_dependent_value=deps,
            ),
        ],
        temperature=500,
        settings=F.Settings(
            atol=1e-12,
            rtol=1e-12,
            transient=True,
            final_time=final_time,
            stepsize=dt,
        ),
    )

    model.initialise()
    model.show_progress_bar = False

    totals = []
    while model.t.value < final_time:
        model.iterate()
        totals.append(integrate(H_gamma, gamma) + integrate(H_lambda, lam))

    c_gamma = H_gamma.subdomain_to_post_processing_solution[gamma].x.array
    c_lambda = H_lambda.subdomain_to_post_processing_solution[lam].x.array
    return np.array(totals), c_gamma, c_lambda


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_nested_exchange_conserves_mass():
    """The exchange moves particles between Gamma and Lambda without creating any."""
    totals, _, _ = run_exchange()
    assert np.allclose(totals, 1.0, atol=1e-8), (totals.min(), totals.max())


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_nested_exchange_reaches_equilibrium():
    """Both fields relax to the same uniform value, fixed by the two extents."""
    _, c_gamma, c_lambda = run_exchange()
    expected = 1.0 / (1.0 + 1.0)  # area(Gamma) / (area(Gamma) + length(Lambda))
    assert np.allclose(c_gamma, expected, atol=1e-3), (c_gamma.min(), c_gamma.max())
    assert np.allclose(c_lambda, expected, atol=1e-3), (c_lambda.min(), c_lambda.max())


# ---------------------------------------------------------------------------
# 2. Lambda's own operator
# ---------------------------------------------------------------------------


def run_diffusion_along_lambda(n=8, D=0.7, decay=2.0):
    """Diffusion along Lambda alone, against a manufactured solution.

    Omega and Gamma are driven to zero and the exchange is switched off, so this
    isolates the terms integrated over Lambda's own submesh -- the tangential gradient
    in particular. The solution is::

        c(x) = cos(pi x),   -D c'' + decay * c = (D pi^2 + decay) cos(pi x)

    chosen so that ``c'(0) = c'(1) = 0``: the natural condition at the ends of Lambda
    holds exactly, which matters because strong boundary conditions on the boundary of
    a nested subdomain are not supported.
    """
    mesh = cube(n)
    omega, gamma, lam = geometry(mesh, D_lambda=D)
    outer = F.SurfaceSubdomain(id=OUTER_ID, locator=lambda x: np.full(x.shape[1], True))

    H_omega = F.Species("c_omega", subdomains=[omega])
    H_gamma = F.Species("c_gamma", subdomains=[gamma])
    H_lambda = F.Species("c_lambda", subdomains=[lam])

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_omega, H_gamma, H_lambda],
        subdomains=[omega, gamma, lam, outer],
        reactions=[
            # a decay on Gamma and on Lambda, so that both steady problems are
            # non-singular under purely natural boundary conditions
            F.Reaction(reactant=[H_gamma], k_0=1.0, E_k=0.0, volume=gamma),
            F.Reaction(reactant=[H_lambda], k_0=decay, E_k=0.0, volume=lam),
        ],
        sources=[
            F.ParticleSource(
                value=lambda x: (D * np.pi**2 + decay) * ufl.cos(np.pi * x[0]),
                species=H_lambda,
                volume=lam,
            ),
        ],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=outer, value=0.0, species=H_omega),
        ],
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )

    model.initialise()
    model.run()

    c = H_lambda.subdomain_to_post_processing_solution[lam]
    x = c.function_space.tabulate_dof_coordinates()[:, 0]
    return c.x.array, np.cos(np.pi * x)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_diffusion_along_nested_subdomain():
    """The tangential operator on Lambda converges to the manufactured solution."""
    errors = []
    for n in (8, 16, 32):
        computed, exact = run_diffusion_along_lambda(n)
        errors.append(np.linalg.norm(computed - exact) / np.linalg.norm(exact))

    rates = [np.log(e0 / e1) / np.log(2.0) for e0, e1 in pairwise(errors)]
    assert all(r > 1.8 for r in rates), (errors, rates)


# ---------------------------------------------------------------------------
# 3. a strong boundary condition at the end of Lambda
# ---------------------------------------------------------------------------


def run_dirichlet_end_of_lambda(n=8, D=0.7, source=3.0):
    """A zero Dirichlet condition at one end of Lambda, with a uniform source.

    The end of Lambda is a point in a 3D mesh, ie. codimension 3: a ``SurfaceSubdomain``
    with ``dim=mesh_dim - 3``. It is located on Lambda's submesh and resolved to Lambda
    through the species of the boundary condition, exactly as a codim-2 surface is
    resolved to the manifold it bounds.

    Omega and Gamma are driven to zero and there is no exchange, so Lambda solves::

        -D c'' = source,  c(0) = 0,  c'(1) = 0   =>   c(x) = source/D * (x - x^2/2)

    The natural condition at the far end holds by construction. In 1D with constant
    coefficients the P1 Galerkin solution is nodally exact, so this is checked to solver
    tolerance rather than to a discretisation error.
    """
    mesh = cube(n)
    omega, gamma, lam = geometry(mesh, D_lambda=D)
    outer = F.SurfaceSubdomain(id=OUTER_ID, locator=lambda x: np.full(x.shape[1], True))
    # codim 3: Lambda is the line y = z = 1/2, so it runs along x and its ends are the
    # points x = 0 and x = 1 of its own submesh
    lam_start = F.SurfaceSubdomain(
        id=END_ID, dim=0, locator=lambda x: np.isclose(x[0], 0.0)
    )

    H_omega = F.Species("c_omega", subdomains=[omega])
    H_gamma = F.Species("c_gamma", subdomains=[gamma])
    H_lambda = F.Species("c_lambda", subdomains=[lam])

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_omega, H_gamma, H_lambda],
        subdomains=[omega, gamma, lam, outer, lam_start],
        reactions=[F.Reaction(reactant=[H_gamma], k_0=1.0, E_k=0.0, volume=gamma)],
        sources=[F.ParticleSource(value=source, species=H_lambda, volume=lam)],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=outer, value=0.0, species=H_omega),
            F.FixedConcentrationBC(subdomain=lam_start, value=0.0, species=H_lambda),
        ],
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )

    model.initialise()
    model.run()

    c = H_lambda.subdomain_to_post_processing_solution[lam]
    x = c.function_space.tabulate_dof_coordinates()[:, 0]
    return c.x.array, source / D * (x - 0.5 * x**2)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_dirichlet_at_the_end_of_a_nested_subdomain():
    """A codim-3 surface pins the value at one end of Lambda and nowhere else."""
    computed, exact = run_dirichlet_end_of_lambda()
    assert np.allclose(computed, exact, atol=1e-8), np.abs(computed - exact).max()


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_codim3_surface_needs_a_nested_subdomain():
    """A codim-3 surface with nothing to bound is rejected, not silently ignored."""
    mesh = cube(4)
    omega, gamma, _ = geometry(mesh)
    stray = F.SurfaceSubdomain(
        id=END_ID, dim=0, locator=lambda x: np.isclose(x[0], 0.0)
    )

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[
            F.Species("c_omega", subdomains=[omega]),
            F.Species("c_gamma", subdomains=[gamma]),
        ],
        subdomains=[omega, gamma, stray],
        temperature=500,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )
    with pytest.raises(ValueError, match="no such subdomain was declared"):
        model.initialise()


# ---------------------------------------------------------------------------
# 4. validation
# ---------------------------------------------------------------------------


def test_nested_subdomain_needs_a_parent():
    """A codim-2 volume subdomain without a parent is rejected, not silently ignored."""
    orphan = F.VolumeSubdomain(
        id=LAMBDA_ID,
        material=F.Material(D_0=1.0, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[1], 0.5),
    )
    with pytest.raises(ValueError, match="parent="):
        orphan.codim(3)


def test_parent_only_on_a_nested_subdomain():
    """A parent on a codim-0 or codim-1 subdomain is a modelling mistake."""
    gamma = F.VolumeSubdomain(id=GAMMA_ID, material=F.Material(D_0=1.0, E_D=0.0), dim=2)
    confused = F.VolumeSubdomain(
        id=LAMBDA_ID, material=F.Material(D_0=1.0, E_D=0.0), dim=2, parent=gamma
    )
    with pytest.raises(ValueError, match="only meaningful"):
        confused.codim(3)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_nested_subdomain_of_a_2d_mesh_cannot_be_mobile():
    """In a 2D mesh a nested subdomain is a set of points, with nothing to diffuse
    along; the model says so rather than failing inside the form compiler."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    omega = F.VolumeSubdomain(
        id=OMEGA_ID,
        material=F.Material(D_0=1.0, E_D=0.0),
        locator=lambda x: np.full(x.shape[1], True),
    )
    gamma = F.VolumeSubdomain(
        id=GAMMA_ID,
        material=F.Material(D_0=1.0, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[1], 0.5),
    )
    point = F.VolumeSubdomain(
        id=LAMBDA_ID,
        material=F.Material(D_0=1.0, E_D=0.0),
        dim=0,
        parent=gamma,
        locator=lambda x: np.isclose(x[0], 0.5),
    )

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[
            F.Species("c_omega", subdomains=[omega]),
            F.Species("c_gamma", subdomains=[gamma]),
            F.Species("c_point", subdomains=[point]),
        ],
        subdomains=[omega, gamma, point],
        temperature=500,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )
    with pytest.raises(NotImplementedError, match="set of points"):
        model.initialise()
