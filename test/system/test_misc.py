from nis import match
import festim as F
import numpy as np
import pytest


def test_convective_flux():
    sim = F.Simulation()

    sim.mesh = F.MeshFromVertices(np.linspace(0, 1, num=50))

    sim.T = F.HeatTransferProblem(transient=False)

    T_external = 2
    sim.boundary_conditions = [
        F.ConvectiveFlux(h_coeff=10, T_ext=T_external, surfaces=1),
        F.DirichletBC(surfaces=2, value=T_external + 1, field="T"),
    ]

    sim.materials = F.Materials([F.Material(1, D_0=1, E_D=0, thermal_cond=2)])

    sim.exports = F.Exports([F.XDMFExport("T", checkpoint=False)])

    sim.settings = F.Settings(1e-10, 1e-10, transient=False)

    sim.initialise()
    sim.run()

    assert sim.T.T(0) > T_external


def test_error_steady_state_with_stepsize():
    """Checks that an error is raised when a stepsize is given for a steady state simulation"""
    my_model = F.Simulation()
    my_model.mesh = F.MeshFromRefinements(1000, size=1)

    my_model.materials = F.Materials([F.Material(D_0=1, E_D=0, id=1)])

    my_model.T = F.Temperature(value=400)

    my_model.settings = F.Settings(
        transient=False,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
    )

    my_model.dt = F.Stepsize(initial_value=1)
    with pytest.raises(
        AttributeError, match="dt must be None in steady state simulations"
    ):
        my_model.initialise()


def test_error_transient_without_stepsize():
    """Checks that an error is raised when a stepsize is not given for a transient simulation"""
    my_model = F.Simulation()
    my_model.mesh = F.MeshFromRefinements(1000, size=1)

    my_model.materials = F.Materials([F.Material(D_0=1, E_D=0, id=1)])

    my_model.T = F.Temperature(value=400)

    my_model.settings = F.Settings(
        transient=True,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
    )

    my_model.dt = None
    with pytest.raises(
        AttributeError, match="dt must be provided in transient simulations"
    ):
        my_model.initialise()


def test_high_recombination_flux():
    """Added test that catches the bug #465
    Checks that with chemical potential and a high recombination coefficient
    the solver doesn't diverge
    """
    model = F.Simulation()

    model.mesh = F.MeshFromVertices(range(10))

    model.boundary_conditions = [
        F.DirichletBC(surfaces=1, value=1, field=0),
        F.RecombinationFlux(Kr_0=1000000, E_Kr=0, order=2, surfaces=2),
    ]

    model.T = F.Temperature(100)

    model.materials = F.Materials([F.Material(id=1, D_0=1, E_D=0, S_0=1, E_S=0)])

    model.settings = F.Settings(1e-10, 1e-10, transient=False, chemical_pot=True)

    model.initialise()
    model.run()
