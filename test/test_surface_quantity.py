import os

from mpi4py import MPI

import basix
import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import (
    create_mesh,
    create_rectangle,
    locate_entities,
    meshtags,
)

import festim as F


def test_surface_flux_export_compute():
    """Test that the surface flux export computes the correct value"""

    # BUILD
    L = 4.0
    D = 1.5
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=4)

    # define mesh ds measure
    facet_indices = np.array(
        dummy_surface.locate_boundary_facet_indices(my_mesh.mesh),
        dtype=np.int32,
    ).flatten()
    tags_facets = np.array(
        [1],
        dtype=np.int32,
    ).flatten()
    facet_meshtags = meshtags(my_mesh.mesh, 0, facet_indices, tags_facets)
    ds = ufl.Measure("ds", domain=my_mesh.mesh, subdomain_data=facet_meshtags)

    # give function to species
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: 2 * x[0] ** 2 + 1)

    my_species = F.Species("H")
    my_species.solution = c

    my_export = F.SurfaceFlux(
        field=my_species,
        surface=dummy_surface,
    )
    my_export.D = D

    # RUN
    my_export.compute(my_species.solution, ds=ds)

    # TEST
    # flux = -D grad(c)_ \cdot n = -D dc/dx = -D * 4 * x
    expected_value = -D * 4 * dummy_surface.x
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)


@pytest.mark.parametrize("value", ["my_export.csv", "my_export.txt"])
def test_title_generation(tmp_path, value):
    """Test that the title is made to be written to the header in a csv or txt file"""
    my_export = F.SurfaceFlux(
        filename=os.path.join(tmp_path, f"{value}"),
        field=F.Species("TEST"),
        surface=F.SurfaceSubdomain1D(id=35, x=1),
    )
    my_export.value = 2.0
    my_export.write(0)
    title = np.genfromtxt(my_export.filename, delimiter=",", max_rows=1, dtype=str)

    expected_title = "TEST flux surface 35"

    assert title[1] == expected_title


def test_write_overwrite(tmp_path):
    """Test that the write method overwrites the file if it already exists"""
    filename = os.path.join(tmp_path, "my_export.csv")
    my_export = F.SurfaceFlux(
        filename=filename,
        field=F.Species("test"),
        surface=F.SurfaceSubdomain1D(id=1, x=0),
    )
    my_export.value = 2.0
    my_export.write(0)
    my_export.write(1)

    my_export2 = F.SurfaceFlux(
        filename=filename,
        field=F.Species("test"),
        surface=F.SurfaceSubdomain1D(id=1, x=0),
    )
    my_export2.value = 3.0
    my_export2.write(1)
    my_export2.write(2)
    my_export2.write(3)

    data = np.genfromtxt(filename, delimiter=",", names=True)
    file_length = data.size
    expected_length = 3

    assert file_length == expected_length


def test_filename_setter_raises_TypeError():
    """Test that a TypeError is raised when the filename is not a string"""

    with pytest.raises(TypeError, match="filename must be of type str"):
        F.SurfaceQuantity(
            filename=1,
            field=F.Species("test"),
            surface=F.SurfaceSubdomain1D(id=1, x=0),
        )


def test_filename_setter_raises_ValueError(tmp_path):
    """Test that a ValueError is raised when the filename does not end with .csv or .txt"""

    with pytest.raises(ValueError):
        F.SurfaceQuantity(
            filename=os.path.join(tmp_path, "my_export.xdmf"),
            field=F.Species("test"),
            surface=F.SurfaceSubdomain1D(id=1, x=0),
        )


def test_field_setter_raises_TypeError():
    """Test that a TypeError is raised when the field is not a F.Species"""

    with pytest.raises(TypeError):
        F.SurfaceQuantity(
            field=1,
            surface=F.SurfaceSubdomain1D(id=1, x=0),
        )


@pytest.mark.parametrize("value", ["my_export.csv", "my_export.txt"])
def test_writer(tmp_path, value):
    """Test that the writes values at each timestep to either a csv or txt file"""
    my_export = F.SurfaceFlux(
        filename=os.path.join(tmp_path, f"{value}"),
        field=F.Species("test"),
        surface=F.SurfaceSubdomain1D(id=1, x=0),
    )
    my_export.value = 2.0

    for i in range(10):
        my_export.write(i)
        file_length = len(np.genfromtxt(my_export.filename, delimiter=","))

        expected_length = i + 2

        assert file_length == expected_length


def test_surface_setter_raises_TypeError():
    """Test that a TypeError is raised when the surface is not a
    F.SurfaceSubdomain"""

    with pytest.raises(
        TypeError, match="surface should be an int or F.SurfaceSubdomain"
    ):
        F.SurfaceQuantity(
            field=F.Species("H"),
            surface="1",
        )


@pytest.mark.parametrize(
    "azimuth_range", [(-1, np.pi), (0, 3 * np.pi), (-1, 3 * np.pi)]
)
def test_azimuthal_range_cylindrical(azimuth_range):
    """
    Tests that an error is raised when the azimuthal range is out of bounds
    """
    with pytest.raises(ValueError):
        F.SurfaceFluxCylindrical("solute", 1, azimuth_range=azimuth_range)


@pytest.mark.parametrize(
    "azimuth_range", [(-1, np.pi), (0, 3 * np.pi), (-1, 3 * np.pi)]
)
def test_azimuthal_range_spherical(azimuth_range):
    """
    Tests that an error is raised when the azimuthal range is out of bounds
    """
    with pytest.raises(ValueError):
        F.SurfaceFluxSpherical("solute", 1, azimuth_range=azimuth_range)


@pytest.mark.parametrize(
    "polar_range", [(0, 2 * np.pi), (-2 * np.pi, 0), (-2 * np.pi, 3 * np.pi)]
)
def test_polar_range_spherical(polar_range):
    """
    Tests that an error is raised when the polar range is out of bounds
    """
    with pytest.raises(ValueError):
        F.SurfaceFluxSpherical("solute", 1, polar_range=polar_range)


@pytest.mark.parametrize("radius", [2, 3, 2.5])
@pytest.mark.parametrize("r0", [0, 2, 1.5])
@pytest.mark.parametrize("height", [2, 3, 5.8])
def test_compute_cylindrical(r0, radius, height):
    """
    Test that SurfaceFluxCylindrical computes the flux correctly on a hollow cylinder
    """
    # creating a mesh with FEniCS
    r1 = r0 + radius
    z0 = 0
    z1 = z0 + height
    mesh_fenics = create_rectangle(
        MPI.COMM_WORLD, np.array([[r0, z0], [r1, z1]]), [10, 10]
    )
    mesh_fenics.topology.create_connectivity(2, 1)
    mesh_fenics.topology.create_connectivity(1, 2)

    class TopSurface(F.SurfaceSubdomain):
        def locate_boundary_facet_indices(self, mesh):
            fdim = mesh.topology.dim - 1
            indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[1], z1))
            return indices

    top_surface = TopSurface(id=1)

    # define mesh ds measure
    num_facets = mesh_fenics.topology.index_map(1).size_local
    mesh_facet_indices = np.arange(num_facets, dtype=np.int32)
    tags_facets = np.full(num_facets, 0, dtype=np.int32)
    entities = top_surface.locate_boundary_facet_indices(mesh_fenics)
    tags_facets[entities] = top_surface.id
    facet_meshtags = meshtags(mesh_fenics, 1, mesh_facet_indices, tags_facets)

    ds = ufl.Measure("ds", domain=mesh_fenics, subdomain_data=facet_meshtags)

    H = F.Species("H")

    D_value = 2
    my_flux = F.SurfaceFluxCylindrical(field=H, surface=top_surface)
    my_flux.D = F.as_fenics_constant(value=D_value, mesh=mesh_fenics)

    V = fem.functionspace(mesh_fenics, ("Lagrange", 2))
    u_expr = lambda x: x[0] ** 2 + x[1] ** 2
    u = fem.Function(V)
    u.interpolate(u_expr)

    expected_value = -2 * np.pi * D_value * z1 * (r1**2 - r0**2)

    my_flux.compute(u=u, ds=ds)
    computed_value = float(my_flux.value)

    assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize("radius", [2, 3, 4.5])
@pytest.mark.parametrize("r0", [0, 2, 1.4])
def test_compute_spherical(r0, radius):
    """
    Test that SurfaceFluxSpherical computes the flux correctly on a hollow sphere
    """
    r1 = r0 + radius

    vertices = np.linspace(r0, r1, 100)
    gdim, shape, degree = 1, "interval", 1
    domain = ufl.Mesh(basix.ufl.element("Lagrange", shape, degree, shape=(gdim,)))
    mesh_points = np.reshape(vertices, (len(vertices), 1))
    indexes = np.arange(mesh_points.shape[0])
    cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)
    mesh_fenics = create_mesh(MPI.COMM_WORLD, cells, mesh_points, domain)
    fdim = mesh_fenics.topology.dim - 1

    class OuterSurface(F.SurfaceSubdomain):
        def locate_boundary_facet_indices(self, mesh):
            fdim = mesh.topology.dim - 1
            indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[0], r1))
            return indices

    outer_surface = OuterSurface(id=1)

    # define mesh ds measure
    num_facets = mesh_fenics.topology.index_map(0).size_local
    mesh_facet_indices = np.arange(num_facets, dtype=np.int32)
    tags_facets = np.full(num_facets, 0, dtype=np.int32)
    entities = outer_surface.locate_boundary_facet_indices(mesh_fenics)
    tags_facets[entities] = outer_surface.id
    facet_meshtags = meshtags(mesh_fenics, fdim, mesh_facet_indices, tags_facets)

    ds = ufl.Measure("ds", domain=mesh_fenics, subdomain_data=facet_meshtags)

    H = F.Species("H")

    D_value = 2
    my_flux = F.SurfaceFluxSpherical(field=H, surface=outer_surface)
    my_flux.D = F.as_fenics_constant(value=D_value, mesh=mesh_fenics)

    u_expr = lambda x: x[0] ** 2
    V = fem.functionspace(mesh_fenics, ("Lagrange", 2))
    u = fem.Function(V)
    u.interpolate(u_expr)

    expected_value = -8 * np.pi * D_value * r1**3

    my_flux.compute(u=u, ds=ds)
    computed_value = float(my_flux.value)

    assert np.isclose(computed_value, expected_value)


def test_surf_flux_cylindrical_allow_meshes():
    """A simple test to check cylindrical meshes are the only
    meshes allowed when using AverageVolumeCylindrical"""

    H = F.Species("H")
    surf = F.SurfaceSubdomain(id=3)
    my_export = F.SurfaceFluxCylindrical(H, surf)

    assert my_export.allowed_meshes == ["cylindrical"]


def test_surf_flux_spherical_allow_meshes():
    """A simple test to check cylindrical meshes are the only
    meshes allowed when using AverageVolumeCylindrical"""

    H = F.Species("H")
    surf = F.SurfaceSubdomain(id=5)
    my_export = F.SurfaceFluxSpherical(H, surf)

    assert my_export.allowed_meshes == ["spherical"]
