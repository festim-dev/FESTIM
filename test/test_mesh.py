import numpy as np
import festim as F

test_1D_mesh = F.Mesh1D(vertices=np.linspace(0, 1, num=10))


def test_get_fdim_of_mesh_1D():
    fdim = test_1D_mesh.fdim

    assert np.isclose(fdim, 0)


def test_get_vdim_of_mesh_1D():
    vdim = test_1D_mesh.vdim

    assert np.isclose(vdim, 1)
