# ruff: noqa: E402
import logging
import os

from mpi4py import MPI

import pytest

ipp = pytest.importorskip("ipyparallel")

import numpy as np
import pytest
from dolfinx import mesh as fenics_mesh
from dolfinx.io import XDMFFile
from dolfinx.mesh import meshtags

import festim as F

mesh_1D = fenics_mesh.create_unit_interval(MPI.COMM_WORLD, 10)
mesh_2D = fenics_mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
mesh_3D = fenics_mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)

# 1D meshtags
my_surface_meshtags = meshtags(
    mesh_1D,
    0,
    np.array([0, 10], dtype=np.int32),
    np.array([1, 2], dtype=np.int32),
)

num_cells = mesh_1D.topology.index_map(1).size_local
my_volume_meshtags = meshtags(
    mesh_1D,
    1,
    np.arange(num_cells, dtype=np.int32),
    np.full(num_cells, 1, dtype=np.int32),
)


@pytest.fixture(scope="module")
def cluster():
    cluster = ipp.Cluster(engines="mpi", n=2, log_level=logging.ERROR)
    rc = cluster.start_and_connect_sync()
    yield rc
    cluster.stop_cluster_sync()


@pytest.mark.parametrize("mesh", [mesh_1D, mesh_2D, mesh_3D])
def test_get_fdim(mesh):
    my_mesh = F.Mesh(mesh)

    assert my_mesh.fdim == mesh.topology.dim - 1


def test_fdim_changes_when_mesh_changes():
    my_mesh = F.Mesh(mesh=mesh_1D)

    for mesh in [mesh_1D, mesh_2D, mesh_3D]:
        my_mesh.mesh = mesh
        assert my_mesh.fdim == mesh.topology.dim - 1


@pytest.mark.parametrize("mesh", [mesh_1D, mesh_2D, mesh_3D])
def test_get_vdim(mesh):
    my_mesh = F.Mesh(mesh)

    assert my_mesh.vdim == mesh.topology.dim


def test_vdim_changes_when_mesh_changes():
    my_mesh = F.Mesh(mesh=mesh_1D)

    for mesh in [mesh_1D, mesh_2D, mesh_3D]:
        my_mesh.mesh = mesh
        assert my_mesh.vdim == mesh.topology.dim


@pytest.mark.parametrize("mesh", [mesh_1D, mesh_2D, mesh_3D])
def test_meshtags_from_xdmf(tmp_path, mesh):
    """Test that the facet and volume meshtags are read correctly from the mesh XDMF
    files."""
    # create mesh functions
    fdim = mesh.topology.dim - 1
    vdim = mesh.topology.dim

    # create facet meshtags
    facet_indices = []
    for i in range(vdim):
        # add the boundary entities at 0 and 1 in each dimension
        facets_zero = fenics_mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[i], 0)
        )
        facets_one = fenics_mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[i], 1)
        )

        facet_indices += [facets_zero, facets_one]

    facet_tags = []

    for idx, _ in enumerate(facet_indices):
        # add tags for each boundary
        facet_tag = np.full(len(facet_indices[i]), idx + 1, dtype=np.int32)
        facet_tags.append(facet_tag)

    facet_tags = np.array(facet_tags).flatten()
    facet_indices = np.array(facet_indices).flatten()

    facet_meshtags = fenics_mesh.meshtags(mesh, fdim, facet_indices, facet_tags)

    # create volume meshtags
    num_cells = mesh.topology.index_map(vdim).size_local
    mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
    # tag all volumes with 0
    tags_volumes = np.full(num_cells, 0, dtype=np.int32)
    # create 2 volumes for x<0.5 and x>0.5
    volume_indices_left = fenics_mesh.locate_entities(
        mesh,
        vdim,
        lambda x: x[0] <= 0.5,
    )

    volume_indices_right = fenics_mesh.locate_entities(
        mesh,
        vdim,
        lambda x: x[0] >= 0.5,
    )
    tags_volumes[volume_indices_left] = 2
    tags_volumes[volume_indices_right] = 3

    volume_meshtags = fenics_mesh.meshtags(mesh, vdim, mesh_cell_indices, tags_volumes)

    # write files
    surface_file_path = os.path.join(tmp_path, "facets_file.xdmf")
    surface_file = XDMFFile(MPI.COMM_WORLD, surface_file_path, "w")
    surface_file.write_mesh(mesh)
    surface_file.write_meshtags(facet_meshtags, mesh.geometry)

    volume_file_path = os.path.join(tmp_path, "volumes_file.xdmf")
    volume_file = XDMFFile(MPI.COMM_WORLD, volume_file_path, "w")
    volume_file.write_mesh(mesh)
    volume_file.write_meshtags(volume_meshtags, mesh.geometry)

    # read files
    my_model = F.HydrogenTransportProblem(
        mesh=F.MeshFromXDMF(
            volume_file=volume_file_path,
            facet_file=surface_file_path,
            mesh_name="mesh",
            surface_meshtags_name="mesh_tags",
            volume_meshtags_name="mesh_tags",
        )
    )
    my_model.define_meshtags_and_measures()

    # TEST
    assert volume_meshtags.dim == my_model.volume_meshtags.dim
    assert volume_meshtags.values.all() == my_model.volume_meshtags.values.all()
    assert facet_meshtags.dim == my_model.facet_meshtags.dim
    assert facet_meshtags.values.all() == my_model.facet_meshtags.values.all()


@pytest.mark.parametrize("vertices", [[1, 2, 3, 4], [0, 0.1, 0.2, 0.3, 0.4, 0.5]])
def test_mesh_vertices_from_list(vertices):
    """Check that giving vertices as a list is correctly processed and ends up as a
    np.ndarray for the mesh."""
    my_mesh = F.Mesh1D(vertices=vertices)

    assert isinstance(my_mesh.vertices, np.ndarray)
    assert len(my_mesh.vertex_blocks) == 1


@pytest.mark.parametrize(
    "vertices",
    [
        [[0, 0.1, 0.2], [1, 1.1, 1.2]],
        [np.linspace(0, 0.2, 3), np.linspace(1, 1.2, 3)],
        # blocks given in non-ascending order
        [[1, 1.1, 1.2], [0, 0.1, 0.2]],
    ],
)
def test_mesh_vertices_from_list_of_lists(vertices):
    """Check that giving vertices as a list of lists creates one disconnected block
    of cells per sublist."""
    my_mesh = F.Mesh1D(vertices=vertices)

    assert len(my_mesh.vertex_blocks) == 2
    assert np.allclose(my_mesh.vertices, [0, 0.1, 0.2, 1, 1.1, 1.2])

    # 2 cells per block, and none spanning the gap
    num_cells = my_mesh.mesh.topology.index_map(1).size_local
    assert num_cells == 4
    midpoints = fenics_mesh.compute_midpoints(
        my_mesh.mesh, 1, np.arange(num_cells, dtype=np.int32)
    )[:, 0]
    assert not np.any((midpoints > 0.2) & (midpoints < 1))


def test_disconnected_blocks_have_two_boundaries_each():
    """Check that each block of a discontinuous 1D mesh has its own two exterior
    facets."""
    my_mesh = F.Mesh1D(vertices=[np.linspace(0, 1, 11), np.linspace(2, 3, 11)])

    my_mesh.mesh.topology.create_connectivity(0, 1)
    facets = fenics_mesh.exterior_facet_indices(my_mesh.mesh.topology)
    coords = np.sort(my_mesh.mesh.geometry.x[facets][:, 0])

    assert np.allclose(coords, [0, 1, 2, 3])


def test_error_raised_when_blocks_overlap():
    """Test that a ValueError is raised when two blocks of vertices overlap."""
    with pytest.raises(ValueError, match="Blocks of vertices must not overlap"):
        F.Mesh1D(vertices=[[0, 0.5, 1], [0.5, 1.5, 2]])


def test_error_raised_when_block_too_small():
    """Test that a ValueError is raised when a block has less than 2 vertices."""
    with pytest.raises(ValueError, match="at least 2 vertices"):
        F.Mesh1D(vertices=[[0, 0.5, 1], [2]])


def test_check_borders_with_blocks():
    """Test that subdomains are checked block per block."""
    my_mesh = F.Mesh1D(vertices=[np.linspace(0, 1, 11), np.linspace(2, 3, 11)])

    vol_1 = F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=None)
    vol_2 = F.VolumeSubdomain1D(id=2, borders=[0.5, 1], material=None)
    vol_3 = F.VolumeSubdomain1D(id=3, borders=[2, 3], material=None)

    my_mesh.check_borders([vol_1, vol_2, vol_3])

    # a block left uncovered
    with pytest.raises(ValueError, match="borders dont match domain borders"):
        my_mesh.check_borders([vol_1, vol_2])

    # a subdomain spanning the gap between two blocks
    spanning = F.VolumeSubdomain1D(id=4, borders=[1, 2], material=None)
    with pytest.raises(ValueError, match="borders dont match domain borders"):
        my_mesh.check_borders([vol_1, vol_2, spanning, vol_3])


def test_error_raised_when_mesh_is_wrong_type():
    """Test that an TypeError is raised when the mesh is not a dolfinx mesh."""

    with pytest.raises(TypeError, match=r"Mesh must be of type dolfinx.mesh.Mesh"):
        F.Mesh(
            mesh="mesh",
        )


def test_create_1D_mesh_parallel(cluster):
    """Test creating a 1D mesh in parallel using ipyparallel."""

    def create_mesh():
        import numpy as np

        import festim as F

        F.Mesh1D(vertices=np.linspace(0, 1, num=1001))

    query = cluster[:].apply_async(create_mesh)
    query.wait()
    assert query.successful(), query.error


@pytest.mark.parametrize("mesh", [mesh_2D, mesh_3D])
def test_attr_error_with_incompitable_mesh_in_spherical(mesh):
    """Test that an AttributeError is raised when trying to use an incompatible mesh
    with a spherical coordinate system."""

    with pytest.raises(
        AttributeError,
        match="spherical coordinates can be used for one-dimensional domains only",
    ):
        F.Mesh(mesh=mesh, coordinate_system="spherical")


def test_attr_error_with_3D_mesh_in_cylindrical():
    """Test that an AttributeError is raised when trying to use an incompatible mesh
    with a cylindrical coordinate system."""

    with pytest.raises(
        AttributeError,
        match="cylindrical coordinates cannot be used for 3D domains",
    ):
        F.Mesh(mesh=mesh_3D, coordinate_system="cylindrical")


@pytest.mark.parametrize("system", ["cyl", "cart", 1.0, "coucou", mesh_1D])
def test_coordinate_system_setter(system):
    if isinstance(system, str):
        err_msg = "coordinate_system must be one of 'cartesian', 'cylindrical', or 'spherical'"  # noqa: E501
    else:
        err_msg = "coordinate_system must be of type str or CoordinateSystem"
    with pytest.raises(
        ValueError,
        match=err_msg,
    ):
        F.Mesh(mesh=mesh_3D, coordinate_system=system)
