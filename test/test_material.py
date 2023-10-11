import festim as F
import numpy as np
from dolfinx import fem
from pytest import raises
import ufl


def test_define_diffusion_coefficient():
    """Test that the diffusion coefficient is correctly defined"""
    T = 10
    D_0 = 1.2
    E_D = 0.5

    my_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0]))
    my_mat = F.Material(D_0=D_0, E_D=E_D)
    D = my_mat.define_diffusion_coefficient(my_mesh.mesh, T)

    D_analytical = D_0 * np.exp(-E_D / F.k_B / T)

    assert float(D) == D_analytical


def test_diffusion_values_type():
    """Test that the diffusion coefficient values types are correctly
    processed"""
    my_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    my_mesh.generate_mesh()
    values = [int(1), fem.Constant(my_mesh.mesh, 1.0), float(1.0)]

    def model(value):
        my_mat = F.Material(value, value, "1")
        my_mat.define_diffusion_coefficient(
            my_mesh.mesh, fem.Constant(my_mesh.mesh, 1.0)
        )

    for value in values:
        model(value)

    with raises(TypeError):
        x = ufl.SpatialCoordinate(my_mesh.mesh)
        model(2 * x[0])
