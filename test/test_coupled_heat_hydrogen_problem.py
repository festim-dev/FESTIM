import numpy as np
import pytest

import festim as F

test_mesh = F.Mesh1D(vertices=np.linspace(0, 1, 100))
test_mat = F.Material(D_0=1, E_D=0, thermal_conductivity=1, heat_capacity=1, density=1)
test_subdomains = [F.VolumeSubdomain1D(id=1, borders=[0, 1], material=test_mat)]
test_H = F.Species("H")


@pytest.mark.parametrize(
    "object",
    ["coucou", 1, 1.0],
)
def test_error_raised_when_wrong_hydrogen_problem_given(object):
    """Test TypeError is raised when an object that isnt a
    festim.HydrogenTransportProblem object is given to hydrogen_problem"""

    test_heat_problem = F.HeatTransferProblem()

    with pytest.raises(
        TypeError,
        match="hydrogen_problem must be a festim.HydrogenTransportProblem object",
    ):
        F.CoupledTransientHeatTransferHydrogenTransport(
            heat_problem=test_heat_problem, hydrogen_problem=object
        )


@pytest.mark.parametrize(
    "object",
    [
        F.HydrogenTransportProblemDiscontinuous(),
        F.HydrogenTransportProblemDiscontinuousChangeVar(),
    ],
)
def test_error_raised_when_wrong_type_hydrogen_problem_given(object):
    """Test TypeError is raised when an object that isnt a
    festim.HydrogenTransportProblem object is given to hydrogen_problem"""

    test_heat_problem = F.HeatTransferProblem()

    with pytest.raises(
        NotImplementedError,
        match="Coupled heat transfer - hydrogen transport simulations with "
        "HydrogenTransportProblemDiscontinuousChangeVar or"
        "HydrogenTransportProblemDiscontinuous"
        "not currently supported",
    ):
        F.CoupledTransientHeatTransferHydrogenTransport(
            heat_problem=test_heat_problem, hydrogen_problem=object
        )


@pytest.mark.parametrize(
    "object",
    ["coucou", 1, 1.0],
)
def test_error_raised_when_wrong_heat_problem_given(object):
    """Test TypeError is raised when an object that isnt a
    festim.HeatTransferProblem object is given to heat_problem"""

    test_hydrogen_problem = F.HydrogenTransportProblem()

    with pytest.raises(
        TypeError,
        match="heat_problem must be a festim.HeatTransferProblem object",
    ):
        F.CoupledTransientHeatTransferHydrogenTransport(
            heat_problem=object, hydrogen_problem=test_hydrogen_problem
        )


def test_initial_dt_values_are_the_same():
    """Test that the smallest of the stepsize intial_value values given is used in both
    the heat_problem and the hydrogen_problem"""

    test_heat_problem = F.HeatTransferProblem(
        mesh=test_mesh,
        subdomains=test_subdomains,
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=0.5, final_time=5),
    )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=test_subdomains,
        species=[test_H],
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=1.5, final_time=5),
    )

    test_coupled_problem = F.CoupledTransientHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    test_coupled_problem.initialise()

    assert np.isclose(
        float(test_coupled_problem.heat_problem.dt),
        float(test_coupled_problem.hydrogen_problem.dt),
    )


def test_dts_always_the_same():
    """Test that the correct dt value is modified"""

    dt = F.Stepsize(
        1000, growth_factor=1000, cutback_factor=0.9, target_nb_iterations=4
    )

    test_heat_problem = F.HeatTransferProblem(
        mesh=test_mesh,
        subdomains=test_subdomains,
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=dt, final_time=1),
        initial_condition=F.InitialTemperature(500),
    )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=test_subdomains,
        species=[test_H],
        initial_conditions=[F.InitialCondition(value=10, species=test_H)],
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=0.1, final_time=1),
    )

    test_coupled_problem = F.CoupledTransientHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    test_coupled_problem.initialise()

    value_dt_heat_initial = test_heat_problem.dt.value

    test_coupled_problem.run()

    assert np.isclose(value_dt_heat_initial, 0.1)


def test_error_raised_when_final_times_not_the_same():
    """Test that an error is raised when the final time values given in the heat_problem
    and the hydrogen_problem are not the same"""

    test_heat_problem = F.HeatTransferProblem(
        mesh=test_mesh,
        subdomains=test_subdomains,
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=1, final_time=10),
    )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=test_subdomains,
        species=[test_H],
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=1, final_time=5),
    )

    test_coupled_problem = F.CoupledTransientHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    with pytest.raises(
        ValueError,
        match="Final time values in the heat transfer and hydrogen transport model"
        " must be the same",
    ):
        test_coupled_problem.initialise()


def test_error_raised_when_both_problems_not_transient():
    """Test that an error is raised when the heat_problem and hydorgen_problem are not
    set to transient"""

    test_heat_problem = F.HeatTransferProblem(
        settings=F.Settings(atol=1, rtol=1, transient=False),
    )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=1, final_time=4),
    )

    with pytest.raises(
        TypeError,
        match="Both the heat and hydrogen problems must be set to transient",
    ):
        F.CoupledTransientHeatTransferHydrogenTransport(
            heat_problem=test_heat_problem,
            hydrogen_problem=test_hydrogen_problem,
        )
