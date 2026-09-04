"""Anisotropic diffusion: a diffusivity given as a tensor rather than a scalar.

A homogenised polycrystal, a rolled or columnar microstructure, or any hcp
lattice conducts differently along different directions, and the material
property that describes that is a second-rank tensor. ``D_0`` may therefore be
given as a matrix, with ``E_D`` staying a scalar -- one activation energy shared
by every direction, the prefactor carrying the anisotropy.

Manufactured solution, chosen so that it exercises the off-diagonal entries::

    c(x, y) = sin(pi x) sin(pi y)
    -div(D grad c) = f,   f = pi**2 [ (Dxx + Dyy) sin(pi x) sin(pi y)
                                      - 2 Dxy cos(pi x) cos(pi y) ]

``c`` vanishes on the boundary of the unit square, so a single Dirichlet
condition on the whole boundary closes the problem. A *rotated* tensor is used as
well as a diagonal one: with ``Dxy == 0`` the second term of ``f`` disappears and
a formulation that mishandled the off-diagonal entries would still pass.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F

from .tools import error_L2


def rotate(principal, angle):
    """``R diag(principal) R^T`` -- the tensor of a material whose fast axis is at
    ``angle`` to x."""
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return rotation @ np.diag(principal) @ rotation.T


def run(n, D, temperature=500.0, E_D=0.0, as_matrix_D=False):
    """Solve the manufactured problem on an ``n x n`` unit square."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    arrhenius = np.exp(-E_D / (F.k_B * temperature))
    D_lab = np.asarray(D) * arrhenius  # what the form should end up using

    if as_matrix_D:
        material = F.Material(D=D_lab.tolist())
    else:
        material = F.Material(D_0=np.asarray(D).tolist(), E_D=E_D)

    volume = F.VolumeSubdomain(
        id=1, material=material, locator=lambda x: np.full_like(x[0], True, dtype=bool)
    )
    boundary = F.SurfaceSubdomain(
        id=2,
        locator=lambda x: (
            np.isclose(x[0], 0)
            | np.isclose(x[0], 1)
            | np.isclose(x[1], 0)
            | np.isclose(x[1], 1)
        ),
    )
    c = F.Species("c")

    pi = np.pi

    def source(x):
        return pi**2 * (
            (D_lab[0, 0] + D_lab[1, 1]) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
            - 2 * D_lab[0, 1] * ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
        )

    model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        species=[c],
        subdomains=[volume, boundary],
        sources=[F.ParticleSource(value=source, species=c, volume=volume)],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=boundary, value=0.0, species=c)
        ],
        temperature=temperature,
        settings=F.Settings(atol=1e-14, rtol=1e-14, transient=False),
    )
    model.show_progress_bar = False
    model.initialise()
    model.run()

    def exact(x):
        return np.sin(pi * x[0]) * np.sin(pi * x[1])

    return error_L2(c.post_processing_solution, exact)


@pytest.mark.parametrize(
    "D",
    [
        np.diag([1.0, 1.0]),  # isotropic, written as a tensor
        np.diag([5.0, 0.5]),  # aligned with the axes
        rotate([5.0, 0.5], np.pi / 6),  # off-diagonal entries non-zero
        rotate([20.0, 0.2], -0.4),  # strongly anisotropic and rotated
    ],
    ids=["identity", "diagonal", "rotated", "strong"],
)
def test_anisotropic_converges(D):
    """Second-order convergence, whatever the orientation of the tensor."""
    # started fine enough to be in the asymptotic regime: a strongly anisotropic
    # tensor is still at 1.89 on an 8 -> 16 pair, and only settles on 2 beyond that
    refinements = [16, 32, 64]
    errors = [run(n, D) for n in refinements]
    rates = [
        np.log(e0 / e1) / np.log(n1 / n0)
        for e0, e1, n0, n1 in zip(
            errors, errors[1:], refinements, refinements[1:], strict=False
        )
    ]
    assert all(rate > 1.9 for rate in rates), (rates, errors)


def test_tensor_D_0_obeys_arrhenius():
    """``D_0`` as a matrix still carries the scalar activation energy."""
    D_0 = rotate([5.0, 0.5], 0.3)
    E_D, T = 0.2, 400.0
    # the source is built from D_0 * exp(-E_D / kT), so the solution only comes out
    # right if the form applies the same factor to the tensor
    assert run(32, D_0, temperature=T, E_D=E_D) < 1e-2


def test_matrix_given_as_D_matches_D_0():
    """Passing the tensor as ``D`` gives the same answer as passing it as ``D_0``."""
    D = rotate([4.0, 0.7], 0.5)
    assert run(24, D, as_matrix_D=True) == pytest.approx(run(24, D), rel=1e-10)


def permeation(D, n_cells=24):
    """A slab held at 1 and 0 on opposite faces, returning both surface fluxes."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n_cells, n_cells)
    volume = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=np.asarray(D).tolist(), E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    left = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 0))
    right = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 1))
    c = F.Species("c")
    fluxes = [
        F.SurfaceFlux(field=c, surface=left),
        F.SurfaceFlux(field=c, surface=right),
    ]

    model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        species=[c],
        subdomains=[volume, left, right],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=left, value=1.0, species=c),
            F.FixedConcentrationBC(subdomain=right, value=0.0, species=c),
        ],
        temperature=500,
        settings=F.Settings(atol=1e-14, rtol=1e-14, transient=False),
        exports=fluxes,
    )
    model.show_progress_bar = False
    model.initialise()
    model.run()
    return fluxes, c, mesh


def test_anisotropic_surface_flux_diagonal():
    """The outlet flux of a slab, against the analytical value.

    For a *diagonal* tensor ``c = 1 - x`` satisfies the zero-flux condition on the
    unloaded faces as well, so it is the exact solution and the flux through
    ``x = 1`` is exactly ``D_xx`` -- the xx entry alone. (For a rotated tensor it
    is not: zero normal flux on the top and bottom then demands
    ``D_yx dc/dx + D_yy dc/dy = 0``, which a profile varying only along x
    violates, so the solution develops boundary layers and no closed form.)
    """
    D = np.diag([3.0, 0.5])
    fluxes, _, _ = permeation(D)
    assert fluxes[1].value == pytest.approx(D[0, 0], rel=1e-10)


def test_anisotropic_surface_flux_off_diagonal():
    """A rotated tensor: the exported flux against the same integral by hand.

    This is what pins the *matrix product* down. A flux written as
    ``-D * dot(grad(c), n)`` instead of ``-dot(D grad(c), n)`` cannot even be
    assembled for a tensor D, but a formulation that silently dropped the
    off-diagonal entries would still produce a number -- and a wrong one.
    """
    D = rotate([3.0, 0.5], 0.35)
    fluxes, c, mesh = permeation(D)

    n = ufl.FacetNormal(mesh)
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 1.0)
    )
    tags = dolfinx.mesh.meshtags(
        mesh, mesh.topology.dim - 1, np.sort(facets), np.full(len(facets), 1, np.int32)
    )
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=tags)
    D_ufl = ufl.as_matrix(D.tolist())
    by_hand = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(
            -ufl.dot(D_ufl * ufl.grad(c.post_processing_solution), n) * ds(1)
        )
    )
    assert fluxes[1].value == pytest.approx(by_hand, rel=1e-10)
    # and the slab conserves: what goes in comes out
    assert fluxes[0].value + fluxes[1].value == pytest.approx(0.0, abs=1e-10)


def outlet_flux_with_inlet(enforce_weakly, D):
    """Outlet flux of the slab, with the inlet BC enforced strongly or weakly.

    Nitsche's consistency and symmetry terms carry ``D grad(.) . n``, and its
    penalty scales on the normal conductance ``n . D . n``. Written with a scalar
    ``D`` those are the wrong quantities for an anisotropic material, and the
    scheme loses consistency rather than failing outright -- so this compares the
    weak enforcement against the strong one on the same problem.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
    volume = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=D.tolist(), E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    left = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 0))
    right = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 1))
    c = F.Species("c")
    flux = F.SurfaceFlux(field=c, surface=right)

    model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        species=[c],
        subdomains=[volume, left, right],
        boundary_conditions=[
            F.FixedConcentrationBC(
                subdomain=left,
                value=1.0,
                species=c,
                enforce_weakly=enforce_weakly,
                penalty=50.0 if enforce_weakly else None,
            ),
            F.FixedConcentrationBC(subdomain=right, value=0.0, species=c),
        ],
        temperature=500,
        settings=F.Settings(atol=1e-14, rtol=1e-14, transient=False),
        exports=[flux],
    )
    model.show_progress_bar = False
    model.initialise()
    model.run()
    # the strongly enforced run is the reference value the weak one must reproduce
    return flux.value


def test_nitsche_anisotropic_agrees_with_strong():
    """A weakly enforced Dirichlet BC has to handle the tensor too.

    Nitsche's consistency and symmetry terms carry ``D grad(.) . n`` and its
    penalty scales on the normal conductance ``n . D . n``. Written with a scalar
    ``D`` those are the wrong quantities for an anisotropic material, so the weak
    enforcement is compared against the strong one on the same problem.
    """
    D = rotate([3.0, 0.5], 0.35)
    strong = outlet_flux_with_inlet(False, D)
    weak = outlet_flux_with_inlet(True, D)
    assert weak == pytest.approx(strong, rel=2e-3)
