import festim as F
import numpy as np
import pytest

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))


def test_define_diffusion_coefficient():
    """Test that the diffusion coefficient is correctly defined"""
    T, D_0, E_D = 10, 1.2, 0.5
    dum_spe = F.Species("dummy")
    my_mat = F.Material(D_0=D_0, E_D=E_D)
    D = my_mat.get_diffusion_coefficient(
        test_mesh.mesh, T, species=dum_spe, model_species=[dum_spe]
    )

    D_analytical = D_0 * np.exp(-E_D / F.k_B / T)

    assert np.isclose(float(D), D_analytical)


def test_multispecies_dict_strings():
    """Test that the diffusion coefficient is correctly defined when keys are
    strings"""
    T = 500
    D_0_A, D_0_B = 1, 2
    E_D_A, E_D_B = 0.1, 0.2
    A, B = F.Species("A"), F.Species("B")
    spe_list = [A, B]

    my_mat = F.Material(D_0={"A": D_0_A, "B": D_0_B}, E_D={"A": E_D_A, "B": E_D_B})
    D_A = my_mat.get_diffusion_coefficient(
        test_mesh.mesh, T, species=A, model_species=spe_list
    )
    D_B = my_mat.get_diffusion_coefficient(
        test_mesh.mesh, T, species=B, model_species=spe_list
    )

    D = [float(D_A), float(D_B)]

    D_A_analytical = D_0_A * np.exp(-E_D_A / F.k_B / T)
    D_B_analytical = D_0_B * np.exp(-E_D_B / F.k_B / T)

    D_analytical = [D_A_analytical, D_B_analytical]

    assert np.isclose(D, D_analytical).all()


def test_multispecies_dict_objects():
    """Test that the diffusion coefficient is correctly defined when keys are
    festim.Species objects"""
    T = 500
    D_0_A, D_0_B = 1, 2
    E_D_A, E_D_B = 0.1, 0.2

    A = F.Species("A")
    B = F.Species("B")
    spe_list = [A, B]

    my_mat = F.Material(D_0={A: D_0_A, B: D_0_B}, E_D={A: E_D_A, B: E_D_B})
    D_A = my_mat.get_diffusion_coefficient(
        test_mesh.mesh, T, species=A, model_species=spe_list
    )
    D_B = my_mat.get_diffusion_coefficient(
        test_mesh.mesh, T, species=B, model_species=spe_list
    )
    D = [float(D_A), float(D_B)]

    D_A_analytical = D_0_A * np.exp(-E_D_A / F.k_B / T)
    D_B_analytical = D_0_B * np.exp(-E_D_B / F.k_B / T)

    D_analytical = [D_A_analytical, D_B_analytical]

    assert np.isclose(D, D_analytical).all()


def test_multispecies_dict_objects_and_strings():
    """Test that the diffusion coefficient is correctly defined when keys
    are a mix of festim.Species objects and strings"""
    T = 500
    D_0_A, D_0_B = 1, 2
    E_D_A, E_D_B = 0.1, 0.2

    A = F.Species("A")
    B = F.Species("B")
    spe_list = [A, B]

    my_mat = F.Material(D_0={A: D_0_A, "B": D_0_B}, E_D={A: E_D_A, "B": E_D_B})
    D_A = my_mat.get_diffusion_coefficient(
        test_mesh.mesh, T, species=A, model_species=spe_list
    )
    D_B = my_mat.get_diffusion_coefficient(
        test_mesh.mesh, T, species=B, model_species=spe_list
    )
    D = [float(D_A), float(D_B)]

    D_A_analytical = D_0_A * np.exp(-E_D_A / F.k_B / T)
    D_B_analytical = D_0_B * np.exp(-E_D_B / F.k_B / T)

    D_analytical = [D_A_analytical, D_B_analytical]

    assert np.isclose(D, D_analytical).all()


def test_multispecies_dict_different_keys():
    """Test that a value error is rasied when the keys of the D_0 and E_D
    are not the same"""
    A = F.Species("A")
    spe_list = [A]
    my_mat = F.Material(D_0={"A": 1, "B": 2}, E_D={"A": 0.1, "B": 0.2, "C": 0.3})

    with pytest.raises(ValueError, match="D_0 and E_D have different keys"):
        my_mat.get_diffusion_coefficient(
            test_mesh.mesh, 500, species=A, model_species=spe_list
        )


def test_multispecies_dict_wrong_name_species_not_found():
    """Test that a value error is rasied when the length of the D_0 and E_D
    are not the same"""
    J = F.Species("J")
    spe_list = [J]
    my_mat = F.Material(D_0={"A": 1, "B": 2}, E_D={"A": 0.1, "B": 0.2})

    with pytest.raises(ValueError, match="Species A not found in list of species"):
        my_mat.get_diffusion_coefficient(
            test_mesh.mesh, 500, species=J, model_species=spe_list
        )


def test_multispecies_dict_contains_species_not_in_species_list():
    """Test that a value error is rasied in the get_diffusion_coefficient
    function"""
    J = F.Species("J")
    A = F.Species("A")
    spe_list = [J]
    my_mat = F.Material(D_0={A: 1, "B": 2}, E_D={A: 0.1, "B": 0.2})

    with pytest.raises(ValueError, match="Species A not found in model species"):
        my_mat.get_diffusion_coefficient(
            test_mesh.mesh, 500, species=J, model_species=spe_list
        )


def test_contains_species_not_in_species_list():
    """Test that a value error is rasied in the get_diffusion_coefficient
    function with one species"""
    J = F.Species("J")
    A = F.Species("A")
    spe_list = [J]
    my_mat = F.Material(D_0=1, E_D=0)

    with pytest.raises(ValueError, match="Species A not found in model species"):
        my_mat.get_diffusion_coefficient(
            test_mesh.mesh, 500, species=A, model_species=spe_list
        )


def test_D_0_type_rasies_error():
    """Test that a value error is rasied in the get_diffusion_coefficient
    function"""
    A = F.Species("A")
    spe_list = [A]
    my_mat = F.Material(D_0=[1, 1], E_D=0.1)

    with pytest.raises(ValueError, match="D_0 and E_D must be either floats or dicts"):
        my_mat.get_diffusion_coefficient(
            test_mesh.mesh, 500, species=A, model_species=spe_list
        )
