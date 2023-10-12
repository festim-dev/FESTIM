import festim as F
import numpy as np

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))


def test_define_diffusion_coefficient():
    """Test that the diffusion coefficient is correctly defined"""
    T, D_0, E_D = 10, 1.2, 0.5

    my_mat = F.Material(D_0=D_0, E_D=E_D)
    D = my_mat.get_diffusion_coefficient(test_mesh.mesh, T)

    D_analytical = D_0 * np.exp(-E_D / F.k_B / T)

    assert np.isclose(float(D), D_analytical)
