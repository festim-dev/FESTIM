import festim as F
from dolfinx import mesh as fenics_mesh
from mpi4py import MPI
import pytest

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
