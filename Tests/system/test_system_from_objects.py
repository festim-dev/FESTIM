import FESTIM
import fenics
import pytest
import sympy as sp
import numpy as np
from pathlib import Path
import timeit


# System tests

def test_run_temperature_stationary(tmpdir):
    '''
    Check that the temperature module works well in 1D stationary
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + 2*FESTIM.x**2
    size = 1

    my_materials = FESTIM.Materials(
        [
            FESTIM.Material(id=1, D_0=4.1e-7, E_D=0.39, thermal_cond=1)
        ]
    )
    my_traps = FESTIM.Traps([])
    my_mesh = FESTIM.MeshFromRefinements(200, size=size)
    my_boundary_conditions = [
        FESTIM.DirichletBC("dc", value=1, component=0, surfaces=[1]),
        FESTIM.DirichletBC("dc", value=u, component="T", surfaces=[1, 2])
    ]

    my_sources = [FESTIM.Source(-4, 1, "T")]
    my_temperature = FESTIM.Temperature("solve_stationary")
    my_settings = FESTIM.Settings(
        absolute_tolerance=1e10, relative_tolerance=1e-9,
        maximum_iterations=50,
        final_time=30
    )
    my_stepsize = FESTIM.Stepsize(initial_value=0.5, stepsize_change_ratio=1, dt_min=1e-5)

    my_derived_quantities = FESTIM.DerivedQuantities(file="derived_quantities.csv", folder=str(Path(d)))
    my_derived_quantities.derived_quantities = [FESTIM.TotalVolume("solute", 1)]

    my_exports = FESTIM.Exports(
        [
            FESTIM.XDMFExports(fields=['T', 'solute'], labels=["temperature", "solute"], folder=str(Path(d))),
            my_derived_quantities,
            FESTIM.Error("T", exact_solution=u)
        ]
    )

    my_sim = FESTIM.Simulation(
        mesh=my_mesh, materials=my_materials,
        boundary_conditions=my_boundary_conditions, traps=my_traps,
        sources=my_sources,
        dt=my_stepsize, settings=my_settings,
        temperature=my_temperature, exports=my_exports)
    my_sim.initialise()
    output = my_sim.run()
    assert output["error"][0] < 1e-9
