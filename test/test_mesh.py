import festim as F
from dolfinx import mesh as fenics_mesh
from mpi4py import MPI
import pytest
import numpy as np
import os
from dolfinx.io import XDMFFile

mesh_1D = fenics_mesh.create_unit_interval(MPI.COMM_WORLD, 10)
mesh_2D = fenics_mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
mesh_3D = fenics_mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)


@pytest.mark.parametrize("mesh", [mesh_1D, mesh_2D, mesh_3D])
def test_get_fdim(mesh):
    my_mesh = F.Mesh(mesh)

    assert my_mesh.fdim == mesh.topology.dim - 1


def test_fdim_changes_when_mesh_changes():
    my_mesh = F.Mesh()

    for mesh in [mesh_1D, mesh_2D, mesh_3D]:
        my_mesh.mesh = mesh
        assert my_mesh.fdim == mesh.topology.dim - 1


@pytest.mark.parametrize("mesh", [mesh_1D, mesh_2D, mesh_3D])
def test_get_vdim(mesh):
    my_mesh = F.Mesh(mesh)

    assert my_mesh.vdim == mesh.topology.dim


def test_vdim_changes_when_mesh_changes():
    my_mesh = F.Mesh()

    for mesh in [mesh_1D, mesh_2D, mesh_3D]:
        my_mesh.mesh = mesh
        assert my_mesh.vdim == mesh.topology.dim


@pytest.mark.parametrize("mesh", [mesh_1D, mesh_2D, mesh_3D])
def test_meshtags_from_xdmf(tmp_path, mesh):
    """Test that the facet and volume meshtags are read correctly from the mesh XDMF files"""
    # create mesh functions
    fdim = mesh.topology.dim - 1
    vdim = mesh.topology.dim

    # create facet meshtags
    facet_indices = []
    for i in range(1):
        facet_indices.append(
            fenics_mesh.locate_entities_boundary(
                mesh, fdim, lambda x: np.isclose(x[i], 0)
            )
        )
        facet_indices.append(
            fenics_mesh.locate_entities_boundary(
                mesh, fdim, lambda x: np.isclose(x[i], 1)
            )
        )
    facet_tags = []
    for i in range(2):
        facet_tags.append(np.full(len(facet_indices[0]), i + 1, dtype=np.int32))

    facet_meshtags = fenics_mesh.meshtags(mesh, fdim, facet_indices, facet_tags)

    # create volume meshtags
    num_cells = mesh.topology.index_map(vdim).size_local
    mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
    tags_volumes = np.full(num_cells, 0, dtype=np.int32)
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
    surface_file = XDMFFile(
        MPI.COMM_WORLD, os.path.join(tmp_path, "facets_file.xdmf"), "w"
    )
    surface_file.write_mesh(mesh)
    surface_file.write_meshtags(facet_meshtags, mesh.geometry)

    volume_file = XDMFFile(
        MPI.COMM_WORLD, os.path.join(tmp_path, "volumes_file.xdmf"), "w"
    )
    volume_file.write_mesh(mesh)
    volume_file.write_meshtags(volume_meshtags, mesh.geometry)

    # read files
    my_mesh = F.MeshFromXDMF(
        volume_file=os.path.join(tmp_path, "volumes_file.xdmf"),
        facet_file=os.path.join(tmp_path, "facets_file.xdmf"),
        mesh_name="mesh",
        meshtags_name="mesh_tags",
    )
    facet_meshtags_2 = my_mesh.define_surface_markers()
    volume_meshtags_2 = my_mesh.define_volume_markers()

    # # TEST
    assert volume_meshtags.dim == volume_meshtags_2.dim
    assert volume_meshtags.values.all() == volume_meshtags_2.values.all()
    assert facet_meshtags.dim == facet_meshtags_2.dim
    assert facet_meshtags.values.all() == facet_meshtags_2.values.all()
