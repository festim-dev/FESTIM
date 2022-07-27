import sympy as sp
import festim
import fenics
from pathlib import Path
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
    v = 1 + sp.cos(2 * fenics.pi * festim.x) * festim.t

    size = 1
    k_0 = 2
    E_k = 1.5
    p_0 = 3
    E_p = 0.2
    T = 700 + 30 * festim.x
    n_trap = 1
    E_D = 0.1
    D_0 = 2
    k_B = festim.k_B
    D = D_0 * sp.exp(-E_D / k_B / T)
    p = p_0 * sp.exp(-E_p / k_B / T)
    k = k_0 * sp.exp(-E_k / k_B / T)

    f = (
        sp.diff(u, festim.t)
        + sp.diff(v, festim.t)
        - D * sp.diff(u, festim.x, 2)
        - sp.diff(D, festim.x) * sp.diff(u, festim.x)
    )
    g = sp.diff(v, festim.t) + p * v - k * u * (n_trap - v)

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
        my_traps = festim.Traps([festim.Trap(k_0, E_k, p_0, E_p, "mat", n_trap)])

        my_initial_conditions = [
            festim.InitialCondition(field=0, value=u),
            festim.InitialCondition(field=1, value=v),
        ]

        my_mesh = festim.MeshFromRefinements(round(size / h), size)

        my_bcs = [
            festim.DirichletBC(surfaces=[1, 2], value=u, field=0),
            festim.DirichletBC(surfaces=[1, 2], value=v, field=1),
        ]

        my_temp = festim.Temperature(T)

        my_sources = [festim.Source(f, 1, "0"), festim.Source(g, 1, "1")]

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
            traps=my_traps,
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

        computed_v = fenics.project(
            my_sim.traps.traps[0].post_processing_solution, my_sim.V_DG1
        )

        error_u = compute_error(u, computed=computed_u, t=my_sim.t, norm="error_max")
        error_v = compute_error(v, computed=computed_v, t=my_sim.t, norm="error_max")

        return error_u, error_v

    tol_u = 1e-7
    tol_v = 1e-6
    sizes = [1 / 1600]
    dt = 0.1 / 50
    for h in sizes:
        error_max_u, error_max_v = run(h)
        msg = (
            "Maximum error on u is:"
            + str(error_max_u)
            + "\n \
            Maximum error on v is:"
            + str(error_max_v)
            + "\n \
            with h = "
            + str(h)
            + "\n \
            with dt = "
            + str(dt)
        )
        print(msg)
        assert error_max_u < tol_u and error_max_v < tol_v
