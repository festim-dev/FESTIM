import fenics
import FESTIM
import sympy as sp
import numpy as np


def test_run_MMS():
    '''
    Test function run() for several refinements
    '''
    r = FESTIM.x

    u = 1 + r**2 #+ sp.sin(2*fenics.pi*r)
    T = 700 + 30*r
    E_D = 0.1
    D_0 = 2
    k_B = FESTIM.k_B
    D = D_0 * sp.exp(-E_D/k_B/T)
    f = -1/(r**2) * sp.diff(D * r**2 * sp.diff(u, r), r)

    my_materials = FESTIM.Materials(
        [
            FESTIM.Material(id=1, D_0=D_0, E_D=E_D)
        ]
    )

    my_mesh = FESTIM.MeshFromVertices(np.linspace(1, 2, 500), type="spherical")

    my_bcs = [
        FESTIM.DirichletBC(surfaces=[1, 2], value=u, component=0),
    ]

    my_temp = FESTIM.Temperature(T)

    my_sources = [
        FESTIM.Source(f, 1, "0"),
    ]

    my_settings = FESTIM.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=50,
        transient=False
    )

    my_exports = FESTIM.Exports(
        [
            FESTIM.XDMFExport("0", "produced_solution", "results", checkpoint=False)
        ]
    )

    my_sim = FESTIM.Simulation(
        mesh=my_mesh, materials=my_materials,
        boundary_conditions=my_bcs,
        exports=my_exports,
        temperature=my_temp, sources=my_sources, settings=my_settings)

    my_sim.initialise()
    my_sim.run()

    produced_solution = my_sim.h_transport_problem.u
    u_expr = fenics.Expression(sp.printing.ccode(u), degree=3)
    expected_solution = fenics.interpolate(u_expr, my_sim.h_transport_problem.V)
    error_L2 = fenics.errornorm(expected_solution, produced_solution, 'L2')
    assert error_L2 < 1e-7
