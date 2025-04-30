import numpy as np
import pytest
from dolfinx import fem

import festim as F

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))


def test_define_diffusion_coefficient():
    """Test that the diffusion coefficient is correctly defined"""
    T, D_0, E_D = 10, 1.2, 0.5
    dum_spe = F.Species("dummy")
    my_mat = F.Material(D_0=D_0, E_D=E_D)
    D = my_mat.get_diffusion_coefficient(test_mesh.mesh, T, species=dum_spe)

    D_analytical = D_0 * np.exp(-E_D / F.k_B / T)

    assert np.isclose(float(D), D_analytical)


def test_multispecies_dict_strings():
    """Test that the diffusion coefficient is correctly defined when keys are
    strings"""
    T = 500
    D_0_A, D_0_B = 1, 2
    E_D_A, E_D_B = 0.1, 0.2
    A, B = F.Species("A"), F.Species("B")

    my_mat = F.Material(D_0={"A": D_0_A, "B": D_0_B}, E_D={"A": E_D_A, "B": E_D_B})
    D_A = my_mat.get_diffusion_coefficient(test_mesh.mesh, T, species=A)
    D_B = my_mat.get_diffusion_coefficient(test_mesh.mesh, T, species=B)

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

    my_mat = F.Material(D_0={A: D_0_A, B: D_0_B}, E_D={A: E_D_A, B: E_D_B})
    D_A = my_mat.get_diffusion_coefficient(test_mesh.mesh, T, species=A)
    D_B = my_mat.get_diffusion_coefficient(test_mesh.mesh, T, species=B)
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

    my_mat = F.Material(D_0={A: D_0_A, "B": D_0_B}, E_D={A: E_D_A, "B": E_D_B})
    D_A = my_mat.get_diffusion_coefficient(test_mesh.mesh, T, species=A)
    D_B = my_mat.get_diffusion_coefficient(test_mesh.mesh, T, species=B)
    D = [float(D_A), float(D_B)]

    D_A_analytical = D_0_A * np.exp(-E_D_A / F.k_B / T)
    D_B_analytical = D_0_B * np.exp(-E_D_B / F.k_B / T)

    D_analytical = [D_A_analytical, D_B_analytical]

    assert np.isclose(D, D_analytical).all()


def test_multispecies_dict_different_keys():
    """Test that a value error is raised when the keys of the D_0 and E_D
    are not the same"""
    A = F.Species("A")
    my_mat = F.Material(D_0={"A": 1, "B": 2}, E_D={"A": 0.1, "B": 0.2, "C": 0.3})

    with pytest.raises(ValueError, match="D_0 and E_D have different keys"):
        my_mat.get_diffusion_coefficient(test_mesh.mesh, 500, species=A)


def test_D_0_type_raises_error():
    """Test that a value error is raised in the get_diffusion_coefficient
    function"""
    # TODO remove this when material class is updated
    A = F.Species("A")
    my_mat = F.Material(D_0=[1, 1], E_D=0.1)

    with pytest.raises(ValueError, match="D_0 and E_D must be either floats or dicts"):
        my_mat.get_diffusion_coefficient(test_mesh.mesh, 500, species=A)


def test_error_raised_when_species_not_given_with_dict():
    """Test that a value error is raised when a species has not been given in
    the get_diffusion_coefficient function when using a dict for properties"""
    A = F.Species("A")
    B = F.Species("B")
    my_mat = F.Material(D_0={A: 1, B: 1}, E_D={A: 0.1, B: 0.1})

    with pytest.raises(
        ValueError, match="species must be provided if D_0 and E_D are dicts"
    ):
        my_mat.get_diffusion_coefficient(test_mesh.mesh, 500)


def test_error_raised_when_species_not_not_in_D_0_dict():
    """Test that a value error is raised when a species has not been given but
    has no value in the dict"""
    A = F.Species("A")
    B = F.Species("B")
    J = F.Species("J")
    my_mat = F.Material(D_0={A: 1, B: 1}, E_D={A: 0.1, B: 0.1})

    with pytest.raises(ValueError, match="J is not in D_0 keys"):
        my_mat.get_diffusion_coefficient(test_mesh.mesh, 500, species=J)


def test_D_0_raises_ValueError_if_species_not_provided_in_dict():
    """Test that a value error is raised in the get_diffusion_coefficient
    function"""
    # TODO remove this when material class is updated
    A = F.Species("A")
    B = F.Species("B")
    my_mat = F.Material(D_0={A: 1, B: 2}, E_D=1)

    with pytest.raises(ValueError, match="species must be provided if D_0 is a dict"):
        my_mat.get_D_0()


def test_D_0_raises_ValueError_if_species_given_not_in_dict_keys():
    """Test that a value error is raised in the get_diffusion_coefficient
    function"""
    # TODO remove this when material class is updated
    A = F.Species("A")
    B = F.Species("B")
    J = F.Species("J")
    my_mat = F.Material(D_0={A: 1, B: 2}, E_D=1)

    with pytest.raises(ValueError, match="J is not in D_0 keys"):
        my_mat.get_D_0(species=J)


def test_raises_TypeError_when_D_0_is_not_correct_type():
    """Test that a TypeError is raised when D_0 is not a float or a dict"""

    my_mat = F.Material(D_0=[1, 2], E_D=1)

    with pytest.raises(TypeError, match="D_0 must be either a float, int or a dict"):
        my_mat.get_D_0()


def test_E_D_raises_ValueError_if_species_not_provided_in_dict():
    """Test that a value error is raised in the get_diffusion_coefficient
    function"""
    # TODO remove this when material class is updated
    A = F.Species("A")
    B = F.Species("B")
    my_mat = F.Material(D_0=1, E_D={A: 1, B: 2})

    with pytest.raises(ValueError, match="species must be provided if E_D is a dict"):
        my_mat.get_E_D()


def test_E_D_raises_ValueError_if_species_given_not_in_dict_keys():
    """Test that a value error is raised in the get_diffusion_coefficient
    function"""
    # TODO remove this when material class is updated
    A = F.Species("A")
    B = F.Species("B")
    J = F.Species("J")
    my_mat = F.Material(D_0=1, E_D={A: 1, B: 2})

    with pytest.raises(ValueError, match="J is not in E_D keys"):
        my_mat.get_E_D(species=J)


def test_raises_TypeError_when_E_D_is_not_correct_type():
    """Test that a TypeError is raised when E_D is not a float or a dict"""

    my_mat = F.Material(D_0=1, E_D=[1, 2])

    with pytest.raises(TypeError, match="E_D must be either a float, int or a dict"):
        my_mat.get_E_D()


@pytest.mark.parametrize(
    "input_value",
    [
        1.0,
        1,
        "coucou",
        lambda T: 1.0 + T,
    ],
)
def test_raises_TypeError_when_D_is_not_correct_type(input_value):
    """Test that a TypeError is raised when D is not an fem.Function"""

    with pytest.raises(TypeError, match="D must be of type fem.Function"):
        F.Material(D=input_value)


def test_get_diffusion_coefficient_returns_function_when_given_to_D():
    """Test that the diffusion coefficient is correctly defined when D is a
    function"""

    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    D = fem.Function(V)
    D.x.array[:] = 2.0

    my_mat = F.Material(D=D)
    D_out = my_mat.get_diffusion_coefficient()

    assert D_out == D


def test_error_raised_when_D_and_D_0_given():
    """Test that a value error is raised when both D and D_0 are given"""

    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    D_func = fem.Function(V)

    with pytest.raises(
        ValueError,
        match="D_0 and D cannot be set at the same time. Please set only one of them.",
    ):
        F.Material(D=D_func, D_0=1)
