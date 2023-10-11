import festim as F
import numpy as np


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
