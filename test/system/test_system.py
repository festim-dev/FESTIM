import festim
import fenics
import pytest
import sympy as sp
import numpy as np
from pathlib import Path
import timeit


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


# System tests


def test_run_temperature_stationary(tmpdir):
    """
    Check that the temperature module works well in 1D stationary
    """
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + 2 * festim.x**2
    size = 1

    my_materials = [festim.Material(id=1, D_0=4.1e-7, E_D=0.39, thermal_cond=1)]

    my_mesh = festim.MeshFromRefinements(200, size=size)
    my_boundary_conditions = [
        festim.DirichletBC(value=1, field=0, surfaces=[1]),
        festim.DirichletBC(value=u, field="T", surfaces=[1, 2]),
    ]

    my_sources = [festim.Source(-4, 1, "T")]
    my_temperature = festim.HeatTransferProblem(transient=False)
    my_settings = festim.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-9,
        maximum_iterations=50,
        final_time=30,
    )
    my_stepsize = festim.Stepsize(
        initial_value=0.5, stepsize_change_ratio=1, dt_min=1e-5
    )

    my_derived_quantities = festim.DerivedQuantities(
        filename="{}/derived_quantities.csv".format(str(Path(d))),
        derived_quantities=[festim.TotalVolume("solute", 1)],
    )

    my_exports = [
        festim.XDMFExports(
            fields=["T", "solute"],
            labels=["temperature", "solute"],
            folder=str(Path(d)),
            checkpoint=False,
        ),
        my_derived_quantities,
    ]

    my_sim = festim.Simulation(
        mesh=my_mesh,
        materials=my_materials,
        boundary_conditions=my_boundary_conditions,
        sources=my_sources,
        dt=my_stepsize,
        settings=my_settings,
        temperature=my_temperature,
        exports=my_exports,
    )
    my_sim.initialise()
    output = my_sim.run()

    error = compute_error(u, computed=my_sim.T.T, t=my_sim.t, norm="error_max")
    assert error < 1e-9


def test_run_temperature_transient(tmpdir):
    """
    Check that the temperature module works well in 1D transient
    """
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + 2 * festim.x**2 + festim.t
    size = 1

    my_materials = festim.Materials(
        [
            festim.Material(
                id=1,
                D_0=4.1e-7,
                E_D=0.39,
                thermal_cond=1,
                rho=1,
                heat_capacity=1,
                borders=[0, size],
            )
        ]
    )
    my_mesh = festim.MeshFromRefinements(200, size)

    my_bcs = [
        festim.DirichletBC(surfaces=[1], value=1, field=0),
        festim.DirichletBC(surfaces=[1, 2], value=u, field="T"),
    ]

    my_temp = festim.HeatTransferProblem(transient=True, initial_value=u)

    my_sources = [
        festim.Source(
            value=sp.diff(u, festim.t) - sp.diff(u, festim.x, 2), volume=1, field="T"
        )
    ]

    my_settings = festim.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-9,
        maximum_iterations=50,
        transient=True,
        final_time=30,
    )

    my_dt = festim.Stepsize(
        initial_value=0.5,
        stepsize_change_ratio=1,
        t_stop=40,
        stepsize_stop_max=0.5,
        dt_min=1e-5,
    )

    my_exports = festim.Exports(
        [
            festim.XDMFExport(
                "T", "temperature", folder=str(Path(d)), checkpoint=False
            ),
        ]
    )

    my_sim = festim.Simulation(
        mesh=my_mesh,
        materials=my_materials,
        sources=my_sources,
        boundary_conditions=my_bcs,
        dt=my_dt,
        settings=my_settings,
        temperature=my_temp,
        exports=my_exports,
    )
    my_sim.initialise()
    my_sim.run()

    error = compute_error(u, computed=my_sim.T.T, t=my_sim.t, norm="error_max")
    assert error < 1e-9


def test_run_MMS(tmpdir):
    """
    Test function run() for several refinements
    """
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + sp.sin(2 * fenics.pi * festim.x) * festim.t
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
            [festim.Material(name="mat", id=1, D_0=D_0, E_D=E_D)]
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
        )

        my_dt = festim.Stepsize(0.1 / 50)
        my_exports = festim.Exports(
            [
                festim.XDMFExport(
                    "retention", "retention", folder=str(Path(d)), checkpoint=False
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
        error_u = compute_error(
            u,
            computed=my_sim.mobile.post_processing_solution,
            t=my_sim.t,
            norm="error_max",
        )
        error_v = compute_error(
            v,
            computed=my_sim.traps.traps[0].post_processing_solution,
            t=my_sim.t,
            norm="error_max",
        )

        return error_u, error_v

    tol_u = 1e-7
    tol_v = 1e-6
    sizes = [1 / 1600, 1 / 1700]
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


def test_run_MMS_chemical_pot(tmpdir):
    """
    Test function run() with conservation of chemical potential (1 material)
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
            [festim.Material(name="mat", id=1, D_0=D_0, E_D=E_D, S_0=2, E_S=0.1)]
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


def test_run_chemical_pot_mass_balance(tmpdir):
    """
    Simple test checking that the mass balance in ensured when solubility
    increases.
    Creates a model with a constant concentration of mobile (c_m(t=0)=1,
    non-flux conditions at surfaces) with a varying temperature
    """
    d = tmpdir.mkdir("Solution_Test")
    my_materials = festim.Materials(
        [festim.Material(id=1, D_0=1, E_D=0.1, S_0=2, E_S=0.1)]
    )

    my_initial_conditions = [
        festim.InitialCondition(field=0, value=1),
    ]

    my_mesh = festim.MeshFromRefinements(5, 1)

    my_temp = festim.Temperature(700 + 210 * festim.t)

    my_settings = festim.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=50,
        transient=True,
        final_time=100,
        chemical_pot=True,
    )

    my_dt = festim.Stepsize(2)

    total_solute = festim.TotalVolume("solute", 1)
    total_retention = festim.TotalVolume("retention", 1)
    derived_quantities = festim.DerivedQuantities()
    derived_quantities.derived_quantities = [total_solute, total_retention]
    my_exports = festim.Exports(
        [
            festim.XDMFExport(
                "retention", "retention", folder=str(Path(d)), checkpoint=False
            ),
            derived_quantities,
        ]
    )

    my_sim = festim.Simulation(
        mesh=my_mesh,
        materials=my_materials,
        initial_conditions=my_initial_conditions,
        temperature=my_temp,
        settings=my_settings,
        dt=my_dt,
        exports=my_exports,
    )

    my_sim.initialise()
    my_sim.run()
    assert total_solute.compute() == pytest.approx(1)
    assert total_retention.compute() == pytest.approx(1)


def test_run_MMS_soret(tmpdir):
    """
    MMS test with soret effect
    """
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + festim.x**2 + festim.t
    T = 2 + sp.cos(2 * fenics.pi * festim.x) * sp.cos(festim.t)
    E_D = 0
    D_0 = 2
    k_B = festim.k_B
    D = D_0 * sp.exp(-E_D / k_B / T)
    H = -2
    S = 3
    R = festim.R
    f = sp.diff(u, festim.t) - sp.diff(
        (
            D
            * (
                sp.diff(u, festim.x)
                + (H * T + S) * u / (R * T**2) * sp.diff(T, festim.x)
            )
        ),
        festim.x,
    )

    def run(h):
        my_materials = festim.Materials(
            [
                festim.Material(
                    id=1, D_0=D_0, E_D=E_D, H={"free_enthalpy": H, "entropy": S}
                )
            ]
        )
        my_initial_conditions = [
            festim.InitialCondition(field=0, value=u),
        ]

        size = 0.1
        my_mesh = festim.MeshFromRefinements(round(size / h), size)

        my_source = festim.Source(f, 1, "solute")

        my_temp = festim.Temperature(T)

        my_bcs = [
            festim.DirichletBC(surfaces=[1, 2], value=u, field=0),
        ]

        my_settings = festim.Settings(
            absolute_tolerance=1e-10,
            relative_tolerance=1e-9,
            maximum_iterations=50,
            transient=True,
            final_time=0.1,
            soret=True,
        )

        my_dt = festim.Stepsize(0.1 / 50)

        my_exports = festim.Exports(
            [
                festim.XDMFExport(
                    "solute", "solute", folder=str(Path(d)), checkpoint=False
                ),
                festim.XDMFExport("T", "T", folder=str(Path(d)), checkpoint=False),
            ]
        )

        my_sim = festim.Simulation(
            mesh=my_mesh,
            materials=my_materials,
            initial_conditions=my_initial_conditions,
            boundary_conditions=my_bcs,
            sources=[my_source],
            temperature=my_temp,
            settings=my_settings,
            dt=my_dt,
            exports=my_exports,
        )

        my_sim.initialise()
        my_sim.run()
        error_u = compute_error(
            u,
            computed=my_sim.mobile.post_processing_solution,
            t=my_sim.t,
            norm="error_max",
        )

        return error_u

    tol_u = 1e-7
    sizes = [1 / 1000, 1 / 2000]
    for h in sizes:
        error_max_u = run(h)
        msg = (
            "L2 error on u is:"
            + str(error_max_u)
            + "\n \
            with h = "
            + str(h)
        )
        print(msg)
        assert error_max_u < tol_u


def test_run_MMS_steady_state(tmpdir):
    """
    MMS test with one trap at steady state
    """
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + festim.x
    v = 1 + festim.x * 2
    size = 1
    k_0 = 2
    E_k = 1.5
    p_0 = 0.2
    E_p = 0.1
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
            [festim.Material(name="mat", id=1, D_0=D_0, E_D=E_D)]
        )

        my_trap = festim.Trap(k_0, E_k, p_0, E_p, ["mat"], n_trap)

        my_initial_conditions = [
            festim.InitialCondition(field=0, value=u),
            festim.InitialCondition(field=1, value=v),
        ]

        size = 0.1
        my_mesh = festim.MeshFromRefinements(round(size / h), size)

        my_sources = [festim.Source(f, 1, "solute"), festim.Source(g, 1, "1")]

        my_temp = festim.Temperature(T)

        my_bcs = [
            festim.DirichletBC(surfaces=[1, 2], value=u, field=0),
            festim.DirichletBC(surfaces=[1, 2], value=v, field=1),
        ]

        my_settings = festim.Settings(
            absolute_tolerance=1e-10,
            relative_tolerance=1e-9,
            maximum_iterations=50,
            transient=False,
            final_time=0.1,
            traps_element_type="DG",
        )

        my_exports = festim.Exports(
            [
                festim.XDMFExport(
                    "solute", "solute", folder=str(Path(d)), checkpoint=False
                ),
                festim.XDMFExport("1", "1", folder=str(Path(d)), checkpoint=False),
                festim.XDMFExport(
                    "retention", "retention", folder=str(Path(d)), checkpoint=False
                ),
                festim.XDMFExport("T", "T", folder=str(Path(d)), checkpoint=False),
            ]
        )

        my_sim = festim.Simulation(
            mesh=my_mesh,
            materials=my_materials,
            traps=my_trap,
            initial_conditions=my_initial_conditions,
            boundary_conditions=my_bcs,
            sources=my_sources,
            temperature=my_temp,
            settings=my_settings,
            exports=my_exports,
        )

        my_sim.initialise()
        my_sim.run()
        error_u = compute_error(
            u,
            computed=my_sim.mobile.post_processing_solution,
            t=my_sim.t,
            norm="error_max",
        )
        error_v = compute_error(
            v,
            computed=my_sim.traps.traps[0].post_processing_solution,
            t=my_sim.t,
            norm="error_max",
        )

        return error_u, error_v

    tol_u = 1e-10
    tol_v = 1e-7
    sizes = [1 / 1000, 1 / 2000]
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
        )
        print(msg)
        assert error_max_u < tol_u and error_max_v < tol_v


def test_chemical_pot_T_solve_stationary(tmpdir):
    """checks that the chemical potential conservation is well computed with
    type solve_stationary for temperature

    adapted to catch bug described in issue #310
    """
    d = tmpdir.mkdir("Solution_Test")
    my_materials = festim.Materials(
        [festim.Material(id=1, D_0=1, E_D=0.1, S_0=2, E_S=0.2, thermal_cond=1)]
    )
    my_mesh = festim.MeshFromRefinements(10, 1)

    my_temp = festim.HeatTransferProblem(transient=False)
    my_bcs = [
        festim.DirichletBC(surfaces=[1, 2], value=1, field="solute"),
        festim.DirichletBC(surfaces=[1], value=300, field="T"),
        festim.DirichletBC(surfaces=[2], value=300, field="T"),
    ]
    my_settings = festim.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=20,
        chemical_pot=True,
        transient=True,
        final_time=100,
    )
    my_dt = festim.Stepsize(10, stepsize_change_ratio=1.2, dt_min=1e-8)
    my_derived_quantities = festim.DerivedQuantities()
    my_derived_quantities.derived_quantities = [festim.TotalSurface("solute", 2)]
    my_exports = festim.Exports(
        [
            festim.XDMFExport(
                "solute", "solute", folder=str(Path(d)), checkpoint=False
            ),
            my_derived_quantities,
        ]
    )

    my_sim = festim.Simulation(
        mesh=my_mesh,
        materials=my_materials,
        boundary_conditions=my_bcs,
        temperature=my_temp,
        settings=my_settings,
        dt=my_dt,
        exports=my_exports,
    )

    my_sim.initialise()
    my_sim.run()

    assert my_derived_quantities.data[-1][1] == pytest.approx(1)


def test_export_particle_flux_with_chemical_pot(tmpdir):
    """Checks that surface particle fluxes can be computed with conservation
    of chemical potential
    """
    d = tmpdir.mkdir("Solution_Test")
    my_materials = festim.Materials(
        [festim.Material(id=1, D_0=2, E_D=1, S_0=2, E_S=1, thermal_cond=2)]
    )
    my_mesh = festim.MeshFromRefinements(10, 1)

    my_temp = festim.Temperature(300)

    my_settings = festim.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-9,
        chemical_pot=True,
        transient=False,
    )
    my_derived_quantities = festim.DerivedQuantities()
    my_derived_quantities.derived_quantities = [
        festim.SurfaceFlux("solute", 1),
        festim.SurfaceFlux("T", 1),
        festim.TotalVolume("retention", 1),
    ]
    my_exports = festim.Exports(
        [
            festim.XDMFExport(
                "solute", "solute", folder=str(Path(d)), checkpoint=False
            ),
            my_derived_quantities,
        ]
    )

    my_sim = festim.Simulation(
        mesh=my_mesh,
        materials=my_materials,
        temperature=my_temp,
        settings=my_settings,
        exports=my_exports,
    )

    my_sim.initialise()
    my_sim.run()


def test_steady_state_with_2_materials():
    """Runs a sim with several materials and checks that the produced value is
    not zero at the centre
    """
    # build
    my_materials = festim.Materials(
        [
            festim.Material(id=[1, 2], D_0=1, E_D=0),
            festim.Material(id=3, D_0=0.25, E_D=0),
        ]
    )

    N = 16
    mesh = fenics.UnitSquareMesh(N, N)
    vm = fenics.MeshFunction("size_t", mesh, 2, 0)
    sm = fenics.MeshFunction("size_t", mesh, 1, 0)

    tol = 1e-14
    subdomain_1 = fenics.CompiledSubDomain("x[1] <= 0.5 + tol", tol=tol)
    subdomain_2 = fenics.CompiledSubDomain(
        "x[1] >= 0.5 - tol && x[0] >= 0.5 - tol", tol=tol
    )
    subdomain_3 = fenics.CompiledSubDomain(
        "x[1] >= 0.5 - tol && x[0] <= 0.5 + tol", tol=tol
    )
    subdomain_1.mark(vm, 1)
    subdomain_2.mark(vm, 2)
    subdomain_3.mark(vm, 3)

    surfaces = fenics.CompiledSubDomain("on_boundary")
    surfaces.mark(sm, 1)
    my_mesh = festim.Mesh(mesh=mesh, volume_markers=vm, surface_markers=sm)

    my_temp = festim.Temperature(30)
    my_bc = festim.DirichletBC([1], value=0, field=0)
    my_source = festim.Source(1, [1, 2, 3], "solute")

    my_settings = festim.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=5,
        transient=False,
    )

    my_sim = festim.Simulation(
        mesh=my_mesh,
        materials=my_materials,
        sources=[my_source],
        temperature=my_temp,
        settings=my_settings,
        boundary_conditions=[my_bc],
    )

    # run
    my_sim.initialise()
    my_sim.run()

    # test

    assert my_sim.h_transport_problem.u(0.5, 0.5) != 0


def test_steady_state_traps_not_everywhere():
    """Creates a simulation problem with a trap not set in all subdomains runs
    the sim and check that the value is not NaN
    """
    # build
    my_materials = festim.Materials(
        [
            festim.Material(name="mat_1", id=1, D_0=1, E_D=0, borders=[0, 0.25]),
            festim.Material(name="mat_2", id=2, D_0=1, E_D=0, borders=[0.25, 0.5]),
            festim.Material(name="mat_3", id=3, D_0=1, E_D=0, borders=[0.5, 1]),
        ]
    )

    my_mesh = festim.MeshFromRefinements(100, 1)

    my_trap = festim.Trap(1, 0, 1, 0, ["mat_1", "mat_3"], 1)

    my_temp = festim.Temperature(1)
    my_bc = festim.DirichletBC([1], value=1, field=0)

    my_settings = festim.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=5,
        traps_element_type="DG",
        transient=False,
    )

    my_sim = festim.Simulation(
        mesh=my_mesh,
        materials=my_materials,
        traps=my_trap,
        temperature=my_temp,
        settings=my_settings,
        boundary_conditions=[my_bc],
    )

    # run
    my_sim.initialise()
    my_sim.run()
    assert not np.isnan(my_sim.h_transport_problem.u.split()[1](0.5))


def test_no_jacobian_update():
    """Runs a transient sim and with the flag "update_jacobian" set to False."""

    # build
    mat = festim.Material(id=1, D_0=1, E_D=0)
    my_materials = festim.Materials([mat])

    my_mesh = festim.MeshFromRefinements(10, 1)

    my_trap = festim.Trap(1, 0, 1, 0, [mat], 1)

    my_temp = festim.Temperature(1)

    my_settings = festim.Settings(
        final_time=10,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=5,
        update_jacobian=False,
    )

    my_dt = festim.Stepsize(1)

    my_sim = festim.Simulation(
        mesh=my_mesh,
        materials=my_materials,
        dt=my_dt,
        traps=my_trap,
        temperature=my_temp,
        settings=my_settings,
    )

    # run
    my_sim.initialise()
    my_sim.run()


def test_nb_iterations_bewteen_derived_quantities_compute():
    """Checks that "nb_iterations_between_compute" has an influence on the
    number of entries in derived quantities
    """

    def init_sim(nb_it_compute):
        my_materials = festim.Materials([festim.Material(id=1, D_0=1, E_D=0)])
        my_mesh = festim.MeshFromRefinements(10, 1)

        my_temp = festim.Temperature(300)

        my_settings = festim.Settings(
            absolute_tolerance=1e10, relative_tolerance=1e-9, final_time=30
        )

        my_dt = festim.Stepsize(4)

        my_derived_quantities = festim.DerivedQuantities(
            nb_iterations_between_compute=nb_it_compute
        )
        my_derived_quantities.derived_quantities = [
            festim.TotalVolume("retention", 1),
        ]
        my_exports = festim.Exports([my_derived_quantities])

        my_sim = festim.Simulation(
            mesh=my_mesh,
            materials=my_materials,
            temperature=my_temp,
            settings=my_settings,
            exports=my_exports,
            dt=my_dt,
        )

        my_sim.initialise()
        return my_sim

    sim_short = init_sim(10)
    sim_short.run()
    short_derived_quantities = sim_short.exports.exports[0].data

    sim_long = init_sim(1)
    sim_long.run()
    long_derived_quantities = sim_long.exports.exports[0].data

    assert len(long_derived_quantities) > len(short_derived_quantities)


def test_error_steady_state_diverges():
    """Checks that when a sim doesn't converge in steady state, an error is
    raised
    """
    # build
    my_materials = festim.Materials(
        [
            festim.Material(id=1, D_0=1, E_D=1),
        ]
    )

    my_mesh = festim.MeshFromRefinements(10, 1)

    my_temp = festim.Temperature(-1)

    my_settings = festim.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        maximum_iterations=2,
        transient=False,
    )

    my_sim = festim.Simulation(
        mesh=my_mesh, materials=my_materials, temperature=my_temp, settings=my_settings
    )

    # run
    my_sim.initialise()
    with pytest.raises(ValueError) as err:
        my_sim.run()

    assert "The solver diverged" in str(err.value)


def test_completion_tone():
    """Checks that when a completion tone is enabled, sim is not affected"""
    my_model = festim.Simulation(log_level=20)
    my_model.mesh = festim.MeshFromRefinements(10, size=1)
    my_model.materials = festim.Materials([festim.Material(id=1, D_0=1, E_D=0)])
    my_model.T = festim.Temperature(100)
    my_model.boundary_conditions = [
        festim.DirichletBC(surfaces=[1, 2], value=0, field=0),
    ]
    my_stepsize = festim.Stepsize(1, stepsize_change_ratio=1.1, dt_min=1e-8)
    my_model.dt = my_stepsize
    my_model.settings = festim.Settings(
        absolute_tolerance=1e-9,
        relative_tolerance=1e-9,
        final_time=1,
    )
    my_model.initialise()
    my_model.run(completion_tone=True)
