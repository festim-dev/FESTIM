import fenics
import festim
import sympy as sp
import numpy as np


def test_run_MMS():
    """
    Tests that festim produces the correct concentration field in spherical
    coordinates
    """
    r = festim.x

    u = 1 + r**2
    T = 700 + 30 * r
    E_D = 0.1
    D_0 = 2
    k_B = festim.k_B
    D = D_0 * sp.exp(-E_D / k_B / T)
    f = -1 / (r**2) * sp.diff(D * r**2 * sp.diff(u, r), r)

    my_materials = festim.Materials([festim.Material(id=1, D_0=D_0, E_D=E_D)])

    my_mesh = festim.MeshFromVertices(np.linspace(1, 2, 500), type="spherical")

    my_bcs = [
        festim.DirichletBC(surfaces=[1, 2], value=u, field=0),
    ]

    my_temp = festim.Temperature(T)

    my_sources = [
        festim.Source(f, 1, "0"),
    ]

    my_settings = festim.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=50,
        transient=False,
    )

    my_sim = festim.Simulation(
        mesh=my_mesh,
        materials=my_materials,
        boundary_conditions=my_bcs,
        temperature=my_temp,
        sources=my_sources,
        settings=my_settings,
    )

    my_sim.initialise()
    my_sim.run()

    produced_solution = my_sim.h_transport_problem.u
    u_expr = fenics.Expression(sp.printing.ccode(u), degree=3)
    expected_solution = fenics.interpolate(u_expr, my_sim.h_transport_problem.V)
    error_L2 = fenics.errornorm(expected_solution, produced_solution, "L2")
    assert error_L2 < 1e-7


def test_MMS_temperature():
    """
    Tests that festim produces the correct temperature field in spherical
    coordinates
    """
    r = festim.x

    T = 700 + 30 * r
    thermal_cond = 2
    f = -1 / (r**2) * sp.diff(thermal_cond * r**2 * sp.diff(T, r), r)

    my_materials = festim.Materials(
        [festim.Material(id=1, D_0=1, E_D=0, thermal_cond=thermal_cond)]
    )

    my_mesh = festim.MeshFromVertices(np.linspace(1, 2, 500), type="spherical")

    my_bcs = [
        festim.DirichletBC(surfaces=[1, 2], value=T, field="T"),
    ]

    my_temp = festim.HeatTransferProblem(transient=False)

    my_sources = [
        festim.Source(f, 1, "T"),
    ]

    my_settings = festim.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=50,
        transient=False,
    )

    my_sim = festim.Simulation(
        mesh=my_mesh,
        materials=my_materials,
        boundary_conditions=my_bcs,
        temperature=my_temp,
        sources=my_sources,
        settings=my_settings,
    )

    my_sim.initialise()
    my_sim.run()

    produced_solution = my_sim.T.T
    u_expr = fenics.Expression(sp.printing.ccode(T), degree=3)
    expected_solution = fenics.interpolate(u_expr, my_sim.h_transport_problem.V)
    error_L2 = fenics.errornorm(expected_solution, produced_solution, "L2")
    assert error_L2 < 1e-7


def test_MMS_Soret_spherical():
    """
    Tests that festim produces the correct concentration field with the Soret flag
    in spherical coordinates with DirichletBC on the left surface and FluxBC on the right surface
    """

    def grad(u):
        """Computes the gradient of a function u.

        Args:
            u (sympy.Expr): a sympy function

        Returns:
            sympy.Matrix: the gradient of u
        """
        return sp.diff(u, r)

    def div(u):
        """Computes the divergence of a vector field u.

        Args:
            u (sympy.Matrix): a sympy vector field

        Returns:
            sympy.Expr: the divergence of u
        """
        return sp.simplify(sp.diff(r**2 * u, r) / r**2)

    # Create the FESTIM model
    my_model = festim.Simulation()

    my_model.mesh = festim.MeshFromVertices(np.linspace(1, 2, 100), type="spherical")

    # Variational formulation
    r = festim.x

    exact_solution = 1 + 7 * r**2  # exact solution

    T = 300 + 20 * r**2

    D = 2
    Q = lambda T: 4 * festim.k_B * T

    flux = -D * (
        grad(exact_solution) + Q(T) * exact_solution / (festim.k_B * T**2) * grad(T)
    )
    mms_source = div(flux)

    my_model.sources = [
        festim.Source(
            mms_source,
            volume=1,
            field="solute",
        ),
    ]

    my_model.boundary_conditions = [
        festim.DirichletBC(surfaces=1, value=exact_solution, field=0),
        festim.FluxBC(surfaces=2, value=-flux, field=0),
    ]

    my_model.materials = festim.Material(id=1, D_0=D, E_D=0, Q=Q)

    my_model.T = festim.Temperature(T)

    my_model.settings = festim.Settings(
        absolute_tolerance=1e-10, relative_tolerance=1e-10, transient=False, soret=True
    )

    my_model.initialise()
    my_model.run()

    expected_solution = fenics.Expression(sp.printing.ccode(exact_solution), degree=4)
    expected_solution = fenics.project(
        expected_solution, fenics.FunctionSpace(my_model.mesh.mesh, "CG", 1)
    )

    produced_solution = my_model.h_transport_problem.mobile.post_processing_solution
    error_L2 = fenics.errornorm(expected_solution, produced_solution, "L2")
    assert error_L2 < 1e-4
