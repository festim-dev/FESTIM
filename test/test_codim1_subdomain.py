"""Unit tests for codim-1 (manifold) volume subdomains. See issue #1208."""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F
from festim.helpers import solution_on


def unit_square(n=8):
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)


@pytest.mark.parametrize(
    "dim, mesh_dim, expected",
    [(None, 2, 0), (2, 2, 0), (1, 2, 1), (2, 3, 1), (3, 3, 0)],
)
def test_codim(dim, mesh_dim, expected):
    subdomain = F.VolumeSubdomain(id=1, material=F.Material(D_0=1, E_D=0), dim=dim)
    assert subdomain.codim(mesh_dim) == expected


@pytest.mark.parametrize("dim, mesh_dim", [(1, 3), (1, 4), (3, 2)])
def test_codim_out_of_range_raises(dim, mesh_dim):
    """Only codimensions 0 and 1 are supported."""
    subdomain = F.VolumeSubdomain(id=1, material=F.Material(D_0=1, E_D=0), dim=dim)
    with pytest.raises(ValueError, match="codimension"):
        subdomain.codim(mesh_dim)


@pytest.mark.parametrize("dim", [1.5, "1", [1]])
def test_dim_type_validation(dim):
    with pytest.raises(TypeError, match="dim must be an integer"):
        F.VolumeSubdomain(id=1, material=F.Material(D_0=1, E_D=0), dim=dim)


@pytest.mark.parametrize("dim", [0, -1])
def test_dim_must_be_positive(dim):
    with pytest.raises(ValueError, match="strictly positive"):
        F.VolumeSubdomain(id=1, material=F.Material(D_0=1, E_D=0), dim=dim)


def test_codim1_subdomain_is_tagged_in_facet_meshtags():
    """A manifold subdomain marks facets, not cells."""
    mesh = F.Mesh(unit_square())
    omega = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=1, E_D=0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    gamma = F.VolumeSubdomain(
        id=2,
        material=F.Material(D_0=1, E_D=0),
        dim=1,
        locator=lambda x: np.isclose(x[0], 0.0),
    )
    ft, ct = mesh.define_meshtags(
        surface_subdomains=[], volume_subdomains=[omega, gamma]
    )

    assert len(ct.find(gamma.id)) == 0, "manifold subdomain must not tag any cell"
    assert len(ft.find(gamma.id)) > 0, "manifold subdomain must tag facets"
    assert np.all(ct.values == omega.id)


def test_codim1_subdomain_id_clashing_with_a_surface_raises():
    """Manifold subdomains share the facet-tag namespace with surface subdomains."""
    omega = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=1, E_D=0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    gamma = F.VolumeSubdomain(
        id=2,
        material=F.Material(D_0=1, E_D=0),
        dim=1,
        locator=lambda x: np.isclose(x[0], 0.0),
    )
    clashing = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 1.0))

    species = F.Species("H", subdomains=[omega])
    problem = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(unit_square()),
        species=[species],
        subdomains=[omega, gamma, clashing],
        temperature=500,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )
    with pytest.raises(ValueError, match="are not unique"):
        problem.define_meshtags_and_measures()


def test_solution_on_prefers_the_requested_subdomain():
    a = F.VolumeSubdomain(id=1, material=F.Material(D_0=1, E_D=0))
    b = F.VolumeSubdomain(id=2, material=F.Material(D_0=1, E_D=0))
    species = F.Species("H", subdomains=[a, b])
    species.subdomain_to_solution = {a: "sol_a", b: "sol_b"}

    assert solution_on(species, a) == "sol_a"
    assert solution_on(species, b) == "sol_b"


def test_solution_on_reaches_across_meshes_when_unambiguous():
    """A codimensional coupling asks a species for a solution it only has elsewhere."""
    omega = F.VolumeSubdomain(id=1, material=F.Material(D_0=1, E_D=0))
    gamma = F.VolumeSubdomain(id=2, material=F.Material(D_0=1, E_D=0), dim=1)
    species = F.Species("H_om", subdomains=[omega])
    species.subdomain_to_solution = {omega: "sol_omega"}

    assert solution_on(species, gamma) == "sol_omega"


def test_solution_on_ambiguous_raises():
    a = F.VolumeSubdomain(id=1, material=F.Material(D_0=1, E_D=0))
    b = F.VolumeSubdomain(id=2, material=F.Material(D_0=1, E_D=0))
    other = F.VolumeSubdomain(id=3, material=F.Material(D_0=1, E_D=0), dim=1)
    species = F.Species("H", subdomains=[a, b])
    species.subdomain_to_solution = {a: "sol_a", b: "sol_b"}

    with pytest.raises(ValueError, match="ambiguous"):
        solution_on(species, other)


def test_manifold_gradient_uses_the_submesh_measure():
    """The measure chosen for gradient terms must give the tangential gradient.

    For ``g(X) = a.X`` the tangential gradient is exactly ``a - (a.n) n`` on any
    manifold, so ``int |grad g|^2`` has a closed form that can be evaluated on the
    parent mesh alone. On a manifold tilted with respect to the coordinate axes, the
    parent facet measure does *not* reproduce it -- this test pins down that FESTIM
    integrates gradient terms over a measure that does.
    """
    n = 6
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    tdim, fdim = mesh.topology.dim, mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    left = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[0], 0.0)
    )
    # tilt Gamma with respect to every coordinate axis
    mesh.geometry.x[:, 0] += 0.7 * mesh.geometry.x[:, 1] + 0.4 * mesh.geometry.x[:, 2]

    imap = mesh.topology.index_map(fdim)
    tags = np.zeros(imap.size_local + imap.num_ghosts, dtype=np.int32)
    tags[left] = 2
    ft = dolfinx.mesh.meshtags(mesh, fdim, np.arange(len(tags), dtype=np.int32), tags)

    gamma = F.VolumeSubdomain(id=2, material=F.Material(D_0=1, E_D=0), dim=2)
    gamma.create_subdomain(mesh, ft)

    a_np = np.array([1.0, -0.6, 0.3])
    V = dolfinx.fem.functionspace(gamma.submesh, ("Lagrange", 1))
    g = dolfinx.fem.Function(V)
    g.interpolate(lambda X: a_np[0] * X[0] + a_np[1] * X[1] + a_np[2] * X[2])

    a = dolfinx.fem.Constant(mesh, a_np)
    normal = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    reference = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form((ufl.dot(a, a) - ufl.dot(a, normal) ** 2) * ds(2))
    )

    dx_grad = ufl.Measure("dx", domain=gamma.submesh)
    computed = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(ufl.grad(g), ufl.grad(g)) * dx_grad)
    )
    assert np.isclose(computed, reference, rtol=1e-10)

    # ... and the parent facet measure would not have worked
    wrong = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(
            ufl.inner(ufl.grad(g), ufl.grad(g)) * ds(2), entity_maps=[gamma.cell_map]
        )
    )
    assert not np.isclose(wrong, reference, rtol=1e-3)
