import festim
import pytest
import fenics as f


@pytest.mark.parametrize("preconditioner", ["default", "icc"])
def test_create_functions_linear_solver_gmres(preconditioner):
    """
    Checks that the function created by create_functions() has the expected value when an
    alternative linear solver is used with/without a preconditioner rather than the default

    Args:
        preconditioner (str): the preconditioning method
    """

    mesh = festim.MeshFromRefinements(10, size=0.1)

    materials = festim.Materials([festim.Material(id=1, D_0=1, E_D=0, thermal_cond=1)])
    mesh.define_measures(materials)

    bcs = [
        festim.DirichletBC(surfaces=[1, 2], value=1, field="T"),
    ]

    my_problem = festim.HeatTransferProblem(
        transient=False,
        absolute_tolerance=1e-03,
        relative_tolerance=1e-10,
        maximum_iterations=30,
        linear_solver="gmres",
        preconditioner=preconditioner,
    )
    my_problem.boundary_conditions = bcs

    # run
    my_problem.create_functions(materials=materials, mesh=mesh)

    assert my_problem.T(0.05) == pytest.approx(1)


class Test_solve_once_with_custom_solver:
    """
    Checks that a custom newton sovler can be used
    """

    def sim(self):
        """Defines a model"""
        bcs = [
            festim.DirichletBC(surfaces=[1, 2], value=1, field="T"),
        ]

        my_problem = festim.HeatTransferProblem(
            transient=False,
            absolute_tolerance=1e-03,
            relative_tolerance=1e-10,
            maximum_iterations=30,
        )
        my_problem.boundary_conditions = bcs
        return my_problem

    def test_custom_solver(self):
        """Solves the system using the built-in solver and using the f.NewtonSolver"""
        mesh = festim.MeshFromRefinements(10, size=0.1)
        materials = festim.Materials(
            [festim.Material(id=1, D_0=1, E_D=0, thermal_cond=1)]
        )
        mesh.define_measures(materials)

        # solve with the built-in solver
        problem_1 = self.sim()
        problem_1.create_functions(materials=materials, mesh=mesh)

        # solve with the custom solver
        problem_2 = self.sim()
        problem_2.newton_solver = f.NewtonSolver()
        problem_2.newton_solver.parameters["absolute_tolerance"] = (
            problem_1.absolute_tolerance
        )
        problem_2.newton_solver.parameters["relative_tolerance"] = (
            problem_1.relative_tolerance
        )
        problem_2.newton_solver.parameters["maximum_iterations"] = (
            problem_1.maximum_iterations
        )
        problem_2.create_functions(materials=materials, mesh=mesh)

        assert (problem_1.T.vector() == problem_2.T.vector()).all()
