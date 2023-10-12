import festim as F
import numpy as np
from dolfinx import fem
import pytest
import ufl

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
x = ufl.SpatialCoordinate(test_mesh.mesh)


def test_define_diffusion_coefficient():
    """Test that the diffusion coefficient is correctly defined"""
    T, D_0, E_D = 10, 1.2, 0.5

    my_mat = F.Material(D_0=D_0, E_D=E_D)
    D = my_mat.define_diffusion_coefficient(test_mesh.mesh, T)

    D_analytical = D_0 * np.exp(-E_D / F.k_B / T)

    assert float(D) == D_analytical


@pytest.mark.parametrize(
    "test_type", [1, fem.Constant(test_mesh.mesh, 1.0), 1.0, "coucou", 2 * x[0]]
)
def test_diffusion_values_type(test_type):
    """Test that the diffusion coefficient values types are correctly
    processed"""
    my_mat = F.Material(test_type, test_type, "1")
    if isinstance(test_type, (fem.Constant, int, float)):
        my_mat.define_diffusion_coefficient(
            test_mesh.mesh, fem.Constant(test_mesh.mesh, 1.0)
        )
    else:
        with pytest.raises(TypeError):
            my_mat.define_diffusion_coefficient(
                test_mesh.mesh, fem.Constant(test_mesh.mesh, 1.0)
            )
