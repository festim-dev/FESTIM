import FESTIM
import pytest
import fenics as f


def test_create_functions_linear_solver_mumps():
    """Checks that the function created by create_functions() has the expected value when an
    alternative linear solver is used rather than the default"""

    mesh = FESTIM.MeshFromRefinements(10, size=0.1)

    materials = FESTIM.Materials([FESTIM.Material(id=1, D_0=1, E_D=0, thermal_cond=1)])
    mesh.define_measures(materials)

    bcs = [
        FESTIM.DirichletBC(surfaces=[1, 2], value=1, field="T"),
    ]

    my_problem = FESTIM.HeatTransferProblem(
        transient=False,
        absolute_tolerance=1e-03,
        relative_tolerance=1e-10,
        maximum_iterations=30,
        linear_solver="mumps",
    )
    my_problem.boundary_conditions = bcs

    # run
    my_problem.create_functions(materials=materials, mesh=mesh)

    assert my_problem.T(0.05) == pytest.approx(1)
