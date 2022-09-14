import sympy as sp
import festim
import fenics
from pathlib import Path
import pytest
import numpy as np


def compute_error(exact, computed, t, norm):
    exact_sol = fenics.Expression(sp.printing.ccode(exact), degree=4, t=t)

    if norm == "error_max":
        mesh = computed.function_space().mesh()
        vertex_values_u = computed.compute_vertex_values(mesh)
        vertex_values_sol = exact_sol.compute_vertex_values(mesh)
        error_max = np.max(np.abs(vertex_values_u - vertex_values_sol))
        return error_max
    else:
        error_L2 = fenics.errornorm(exact_sol, computed, norm)
        return error_L2


def test_run_MMS_chemical_pot(tmpdir):
    """
    Test function run() with conservation of chemical potential henry
    (1 material)
    """
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + sp.sin(2 * fenics.pi * festim.x) * festim.t + festim.t

    size = 1
    T = 700 + 30 * festim.x
    E_D = 0.1
    D_0 = 2
    k_B = festim.k_B
    D = D_0 * sp.exp(-E_D / k_B / T)

    f = (
        sp.diff(u, festim.t)
        - D * sp.diff(u, festim.x, 2)
        - sp.diff(D, festim.x) * sp.diff(u, festim.x)
    )

    def run(h):
        my_materials = festim.Materials(
            [
                festim.Material(
                    name="mat",
                    id=1,
                    D_0=D_0,
                    E_D=E_D,
                    S_0=2,
                    E_S=0.1,
                    solubility_law="henry",
                )
            ]
        )

        my_initial_conditions = [
            festim.InitialCondition(field=0, value=u),
        ]

        my_mesh = festim.MeshFromRefinements(round(size / h), size)

        my_bcs = [
            festim.DirichletBC(surfaces=[1, 2], value=u, field=0),
        ]

        my_temp = festim.Temperature(T)

        my_sources = [festim.Source(f, 1, "0")]

        my_settings = festim.Settings(
            absolute_tolerance=1e-10,
            relative_tolerance=1e-9,
            maximum_iterations=50,
            transient=True,
            final_time=0.1,
            chemical_pot=True,
        )

        my_dt = festim.Stepsize(0.1 / 50)
        my_exports = festim.Exports(
            [
                festim.TXTExport(
                    "solute", times=[100], label="solute", folder=str(Path(d))
                ),
            ]
        )

        my_sim = festim.Simulation(
            mesh=my_mesh,
            materials=my_materials,
            initial_conditions=my_initial_conditions,
            boundary_conditions=my_bcs,
            temperature=my_temp,
            sources=my_sources,
            settings=my_settings,
            dt=my_dt,
            exports=my_exports,
        )

        my_sim.initialise()
        my_sim.run()

        computed_u = fenics.project(
            my_sim.mobile.post_processing_solution, my_sim.V_DG1
        )

        error_u = compute_error(u, computed=computed_u, t=my_sim.t, norm="error_max")

        return error_u

    tol_u = 1e-7
    sizes = [1 / 1600]
    dt = 0.1 / 50
    for h in sizes:
        error_max_u = run(h)
        msg = (
            "Maximum error on u is:"
            + str(error_max_u)
            + "\n \
            with h = "
            + str(h)
            + "\n \
            with dt = "
            + str(dt)
        )
        print(msg)
        assert error_max_u < tol_u


def test_error_raised_when_henry_and_traps():
    """Checks an error is raised when adding a trap with
    chemical potential in a Henry material"""

    my_sim = festim.Simulation()
    my_sim.materials = festim.Material(
        name="mat",
        id=1,
        D_0=1,
        E_D=0,
        S_0=2,
        E_S=0.1,
        solubility_law="henry",
    )

    my_sim.mesh = festim.MeshFromVertices([0, 1, 2, 3, 4])

    my_sim.traps = festim.Trap(k_0=1, E_k=0, p_0=1, E_p=0, materials="mat", density=1)
    my_sim.T = festim.Temperature(400)

    my_sim.settings = festim.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=50,
        chemical_pot=True,
        transient=False,
    )

    with pytest.raises(NotImplementedError):
        my_sim.initialise()
