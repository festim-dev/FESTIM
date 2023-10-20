import festim as F
import numpy as np

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))


def test_define_diffusion_coefficient():
    """Test that the diffusion coefficient is correctly defined"""
    T, D_0, E_D = 10, 1.2, 0.5

    my_mat = F.Material(D_0=D_0, E_D=E_D)
    D = my_mat.get_diffusion_coefficient(test_mesh.mesh, T, species="dummy")

    D_analytical = D_0 * np.exp(-E_D / F.k_B / T)

    assert np.isclose(float(D), D_analytical)


def test_multispecies_dict():
    T = 500
    D_0_A, D_0_B = 1, 2
    E_D_A, E_D_B = 0.1, 0.2

    my_mat = F.Material(D_0={"A": D_0_A, "B": D_0_B}, E_D={"A": E_D_A, "B": E_D_B})
    D_A = my_mat.get_diffusion_coefficient(test_mesh.mesh, T, species="A")
    D_B = my_mat.get_diffusion_coefficient(test_mesh.mesh, T, species="B")
    D = [float(D_A), float(D_B)]

    D_A_analytical = D_0_A * np.exp(-E_D_A / F.k_B / T)
    D_B_analytical = D_0_B * np.exp(-E_D_B / F.k_B / T)

    D_analytical = [D_A_analytical, D_B_analytical]

    assert np.isclose(D, D_analytical).all()
