"""Unit tests for codim-1 (manifold) volume subdomains. See issue #1208."""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl
from scifem import assemble_scalar

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


def test_codim1_subdomain_not_overlapping_a_surface_is_fine():
    """An interior manifold and an exterior surface both keep their facets."""
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
        locator=lambda x: np.isclose(x[0], 0.5),
    )
    left = F.SurfaceSubdomain(id=5, locator=lambda x: np.isclose(x[0], 0.0))

    ft, _ = mesh.define_meshtags(
        surface_subdomains=[left], volume_subdomains=[omega, gamma]
    )

    assert len(ft.find(gamma.id)) == 8
    assert len(ft.find(left.id)) == 8


def test_codim2_subdomain_tags_no_facet_of_the_parent_mesh():
    """A codim-2 surface is resolved on the manifold's submesh, not in the facet tags,
    so its locator must not claim facets it does not own."""
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
        locator=lambda x: np.isclose(x[0], 0.5),
    )
    # the bottom end of gamma, located with a locator that also matches a whole edge
    tip = F.SurfaceSubdomain(id=7, dim=0, locator=lambda x: np.isclose(x[1], 0.0))

    ft, _ = mesh.define_meshtags(
        surface_subdomains=[tip], volume_subdomains=[omega, gamma]
    )

    assert len(ft.find(tip.id)) == 0


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


@pytest.mark.parametrize(
    "dim, mesh_dim, expected",
    [(None, 2, 1), (1, 2, 1), (0, 2, 2), (1, 3, 2), (2, 3, 1)],
)
def test_surface_codim(dim, mesh_dim, expected):
    """A surface subdomain bounds a volume subdomain: codim 1 for an ordinary one,
    codim 2 for the boundary of a manifold."""
    assert F.SurfaceSubdomain(id=1, dim=dim).codim(mesh_dim) == expected


@pytest.mark.parametrize("dim, mesh_dim", [(0, 3), (3, 2)])
def test_surface_codim_out_of_range_raises(dim, mesh_dim):
    with pytest.raises(ValueError, match="codimension"):
        F.SurfaceSubdomain(id=1, dim=dim).codim(mesh_dim)


@pytest.mark.parametrize("dim", [1.5, "1", [1]])
def test_surface_dim_type_validation(dim):
    with pytest.raises(TypeError, match="dim must be an integer"):
        F.SurfaceSubdomain(id=1, dim=dim)


def test_surface_dim_must_not_be_negative():
    with pytest.raises(ValueError, match="positive"):
        F.SurfaceSubdomain(id=1, dim=-1)


def build_manifold_model(surfaces, species_on=None):
    """A bulk plus a manifold on its top edge, with the given surface subdomains."""
    mesh = unit_square()
    omega = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=1, E_D=0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    gamma = F.VolumeSubdomain(
        id=2,
        material=F.Material(D_0=1, E_D=0),
        dim=1,
        locator=lambda x: np.isclose(x[1], 1.0),
    )
    bottom = F.SurfaceSubdomain(id=9, locator=lambda x: np.isclose(x[1], 0.0))
    H_om = F.Species("H_om", subdomains=[omega])
    H_gam = F.Species("H_gam", subdomains=species_on or [gamma])
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_om, H_gam],
        subdomains=[omega, gamma, bottom, *surfaces],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=bottom, value=0.0, species=H_om)
        ],
        temperature=500,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )
    return model, omega, gamma, H_gam


def test_manifold_boundary_is_not_tagged_and_shares_no_id_namespace():
    """A codim-2 surface carries no facet tag, so its id cannot clash with a manifold's
    and it must not appear in the facet meshtags."""
    # id 2 is deliberately the same as the manifold's: legal, because they live in
    # different namespaces
    end = F.SurfaceSubdomain(id=2, dim=0, locator=lambda x: np.isclose(x[0], 0.0))
    model, _, gamma, H_gam = build_manifold_model([end])
    model.boundary_conditions.append(
        F.FixedConcentrationBC(subdomain=end, value=1.0, species=H_gam)
    )
    model.initialise()

    assert model.manifold_boundary_subdomains == [end]
    assert end not in model.facet_surface_subdomains
    # the facets tagged 2 are the manifold's, and nothing else claims them
    assert len(model.facet_meshtags.find(gamma.id)) > 0
    assert end not in model.surface_to_volume


def test_manifold_boundary_without_a_manifold_raises():
    """A codim-2 surface with nothing to bound would otherwise be silently ignored."""
    end = F.SurfaceSubdomain(id=3, dim=0, locator=lambda x: np.isclose(x[0], 0.0))
    mesh = F.Mesh(unit_square())
    omega = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=1, E_D=0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=mesh,
        species=[F.Species("H", subdomains=[omega])],
        subdomains=[omega, end],
        temperature=500,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )
    with pytest.raises(ValueError, match="no such subdomain was declared"):
        model.initialise()


def test_manifold_boundary_locator_matching_nothing_raises():
    """A locator selecting a point inside the manifold, or none at all, is an error --
    the boundary condition would otherwise silently do nothing."""
    end = F.SurfaceSubdomain(id=3, dim=0, locator=lambda x: np.isclose(x[0], 0.5))
    model, _, _, H_gam = build_manifold_model([end])
    model.boundary_conditions.append(
        F.FixedConcentrationBC(subdomain=end, value=1.0, species=H_gam)
    )
    with pytest.raises(ValueError, match="matched no boundary entity"):
        model.initialise()


def test_manifold_boundary_with_ambiguous_species_raises():
    """The manifold a codim-2 surface bounds comes from the species of its bc, so that
    species must live on exactly one manifold."""
    mesh = unit_square()
    omega = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=1, E_D=0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    gamma_top = F.VolumeSubdomain(
        id=2,
        material=F.Material(D_0=1, E_D=0),
        dim=1,
        locator=lambda x: np.isclose(x[1], 1.0),
    )
    gamma_left = F.VolumeSubdomain(
        id=3,
        material=F.Material(D_0=1, E_D=0),
        dim=1,
        locator=lambda x: np.isclose(x[0], 0.0),
    )
    bottom = F.SurfaceSubdomain(id=9, locator=lambda x: np.isclose(x[1], 0.0))
    end = F.SurfaceSubdomain(id=4, dim=0, locator=lambda x: np.isclose(x[0], 0.0))

    H_om = F.Species("H_om", subdomains=[omega])
    # lives on *both* manifolds, so the surface cannot be resolved
    H_gam = F.Species("H_gam", subdomains=[gamma_top, gamma_left])

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_om, H_gam],
        subdomains=[omega, gamma_top, gamma_left, bottom, end],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=bottom, value=0.0, species=H_om),
            F.FixedConcentrationBC(subdomain=end, value=1.0, species=H_gam),
        ],
        temperature=500,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )
    with pytest.raises(ValueError, match="cannot tell which manifold"):
        model.initialise()


def test_manifold_time_constants_live_on_the_submesh_and_track():
    """``t``, ``dt`` and a constant ``T`` are mirrored onto a manifold's submesh.

    A manifold's self terms are integrated over its submesh, and a ``fem.Constant``
    bound to the parent mesh cannot appear in such an integral -- FFCx fails with an
    ``UnboundLocalError`` that says nothing about the cause. The mirrors are only
    correct if they follow the values they mirror, which is what one step checks.
    """
    mesh = unit_square()
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
    right = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 1.0))
    H_om = F.Species("H_om", subdomains=[omega])
    H_gam = F.Species("H_gam", subdomains=[gamma])

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[H_om, H_gam],
        subdomains=[omega, gamma, right],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=right, value=0.0, species=H_om)
        ],
        temperature=lambda t: 500.0 + 10.0 * t,
        settings=F.Settings(
            atol=1e-10,
            rtol=1e-10,
            transient=True,
            final_time=1.0,
            stepsize=F.Stepsize(initial_value=0.25),
        ),
    )
    model.show_progress_bar = False
    model.initialise()

    for mirror in (gamma.sub_t, gamma.sub_dt, gamma.sub_T):
        assert mirror.ufl_domain() is gamma.submesh.ufl_domain()
    # a codim-0 subdomain has no mirror to keep: it uses the parent-mesh constants
    assert omega.sub_t is None and omega.sub_dt is None

    model.iterate()

    assert float(gamma.sub_t) == float(model.t)
    assert float(gamma.sub_dt) == float(model.dt)
    assert float(gamma.sub_T) == float(model.temperature_fenics)


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


def _three_strips(n=12):
    """A unit square cut into three vertical strips tagged 1, 2, 3, with the facets at
    ``x = 1/3`` (between strips 1 and 2) and ``x = 2/3`` tagged 7 and 8."""
    mesh = unit_square(n)
    tdim = mesh.topology.dim
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
    values = (np.digitize(midpoints[:, 0], [1 / 3, 2 / 3]) + 1).astype(np.int32)
    ct = dolfinx.mesh.meshtags(mesh, tdim, cells, values)

    facets, tags = [], []
    for tag, x0 in ((7, 1 / 3), (8, 2 / 3)):
        found = dolfinx.mesh.locate_entities(
            mesh, tdim - 1, lambda x, x0=x0: np.isclose(x[0], x0)
        )
        facets.append(found)
        tags.append(np.full(len(found), tag, dtype=np.int32))
    order = np.argsort(np.concatenate(facets))
    ft = dolfinx.mesh.meshtags(
        mesh,
        tdim - 1,
        np.concatenate(facets)[order].astype(np.int32),
        np.concatenate(tags)[order],
    )
    return ct, ft


@pytest.mark.parametrize("plus, minus", [(1, 2), (2, 1)])
def test_ordered_interior_facet_data_puts_requested_subdomain_on_plus(plus, minus):
    """The "+" restriction is the subdomain asked for, whichever way DOLFINx (or the
    cell tag values) happens to order the two cells of the facet."""
    ct, ft = _three_strips()
    material = F.Material(D_0=1, E_D=0)
    tag, data = F.subdomain.compute_ordered_interior_facet_data(
        ct,
        ft,
        7,
        F.VolumeSubdomain(id=plus, material=material),
        F.VolumeSubdomain(id=minus, material=material),
    )
    assert tag == 7
    quadruples = data.reshape(-1, 4)
    assert (ct.values[quadruples[:, 0]] == plus).all()
    assert (ct.values[quadruples[:, 2]] == minus).all()


def test_ordered_interior_facet_data_rejects_facets_elsewhere():
    """Facets that do not all separate the two given subdomains cannot be ordered."""
    ct, ft = _three_strips()
    material = F.Material(D_0=1, E_D=0)
    with pytest.raises(ValueError, match="do not all separate volume subdomain"):
        F.subdomain.compute_ordered_interior_facet_data(
            ct,
            ft,
            8,  # between strips 2 and 3
            F.VolumeSubdomain(id=1, material=material),
            F.VolumeSubdomain(id=2, material=material),
        )


def test_surface_quantity_accepts_a_manifold_subdomain():
    """A manifold is passed directly wherever a surface is expected, exports included.

    Which subdomain object it is cannot tell whether it is really codim 1 -- that needs
    the mesh -- so the setter accepts any volume subdomain and the problem checks at
    initialisation.
    """
    gamma = F.VolumeSubdomain(id=2, material=F.Material(D_0=1, E_D=0), dim=1)
    export = F.SurfaceFlux(field=F.Species("H"), surface=gamma)
    assert export.surface is gamma


@pytest.mark.parametrize("bad", [True, "top", 1.0, None])
def test_surface_quantity_rejects_other_types(bad):
    with pytest.raises(TypeError, match="surface should be"):
        F.SurfaceFlux(field=F.Species("H"), surface=bad)


def test_export_volume_measure_is_on_the_submesh_for_a_manifold():
    """A manifold's fields live on its submesh, so a volume quantity over it integrates
    there. The measure carries a tag over the whole submesh, so indexing it by the
    subdomain id -- which is what the export does -- selects all of it."""
    model, omega, gamma, _ = build_manifold_model([])
    model.initialise()

    dx_gamma = model.export_volume_measure(gamma)
    assert dx_gamma.ufl_domain() is gamma.submesh.ufl_domain()
    # the manifold is the full top edge of the unit square
    assert np.isclose(assemble_scalar(1 * dx_gamma(gamma.id)), 1.0)

    # an ordinary subdomain is untouched
    assert model.export_volume_measure(omega) is model.dx


def test_facet_measure_is_unindexed():
    """Derived quantities index the measure they are handed by the id of the subdomain
    they export, exactly as they do with the parent ds, so it must reach them
    unrestricted. A manifold on the boundary of the mesh is integrated with that same
    parent ``ds``."""
    model, _, gamma, _ = build_manifold_model([])
    model.initialise()

    measure = model.facet_measure(gamma)
    assert measure is model.ds
    assert measure.subdomain_id() == "everywhere"
    assert measure(gamma.id).subdomain_id() == gamma.id


def test_manifold_boundary_measure_is_memoised():
    """DOLFINx requires every integral of a compiled form to share the *same*
    subdomain_data object, not merely an equal one."""
    end = F.SurfaceSubdomain(id=7, dim=0, locator=lambda x: np.isclose(x[0], 0.0))
    model, _, gamma, H_gam = build_manifold_model([end])
    model.exports = [F.TotalSurface(field=H_gam, surface=end)]
    model.initialise()

    first = model._manifold_boundary_measure(end, gamma)
    assert model._manifold_boundary_measure(end, gamma) is first
    assert first.ufl_domain() is gamma.submesh.ufl_domain()
    # a single endpoint of a 1D submesh, of unit measure
    assert np.isclose(assemble_scalar(1 * first(end.id)), 1.0)
