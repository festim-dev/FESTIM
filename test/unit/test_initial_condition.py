import festim
import pytest
import numpy


def test_error_initialisation_from_xdmf_missing_time_step():
    """
    Test that the function fails initialise_solutions if
    there's a missing key
    """

    with pytest.raises(ValueError, match=r"time_step"):
        festim.InitialCondition(value="my_file.xdmf", label="my_label", time_step=None)


def test_error_initialisation_from_xdmf_missing_label():
    """
    Test that the function fails initialise_solutions if
    there's a missing key
    """
    with pytest.raises(ValueError, match=r"label"):
        festim.InitialCondition(value="my_file.xdmf", label=None, time_step=1)


def test_temperature_and_DirichletBC():
    """
    Test that the code returns an error when a Dirichlet BC is set with T when there
    is also a Festim.Temperature already set
    """
    my_model = festim.Simulation()

    my_model.T = festim.Temperature(873)

    my_model.mesh = festim.MeshFromVertices(vertices=numpy.linspace(0, 1, num=1000))

    my_model.boundary_conditions = [
        festim.DirichletBC(field="T", value=1200, surfaces=1),
        festim.DirichletBC(field="T", value=373, surfaces=2),
    ]

    my_model.materials = festim.Material(1, 1, 0)

    my_model.settings = festim.Settings(
        absolute_tolerance=1e-10, relative_tolerance=1e-10, transient=False
    )

    with pytest.raises(
        ValueError,
        match="cannot use boundary conditions with Temperature, use HeatTransferProblem instead.",
    ):
        my_model.initialise()
