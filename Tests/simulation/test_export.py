import fenics
import FESTIM
import pytest


def test_output_of_run_without_traps_no_chemical_pot():
    '''
    Test method make_output() returns a dict with the appropriate keys
    without traps nor chemical pot
    '''
    # build
    mesh = fenics.UnitSquareMesh(10, 10)
    V_CG1 = fenics.FunctionSpace(mesh, 'CG', 1)
    V_DG1 = fenics.FunctionSpace(mesh, 'DG', 1)
    my_sim = FESTIM.Simulation(
        {"boundary_conditions": [], "exports": {}, "traps": []})
    my_sim.mesh = FESTIM.Mesh(mesh)
    my_temp = FESTIM.Temperature("solve_stationary")
    my_temp.T = fenics.Function(V_CG1)
    my_sim.T = my_temp
    my_sim.V_DG1 = V_DG1
    my_sim.mobile = FESTIM.Mobile()

    # concentrations
    val_solute = 1

    solute = fenics.project(
        fenics.Expression(str(val_solute), degree=0),
        V_CG1)

    retention_expected = fenics.project(val_solute, V_CG1)
    my_sim.exports = FESTIM.Exports([])
    my_sim.h_transport_problem = FESTIM.HTransportProblem(
        my_sim.mobile, my_sim.traps, my_sim.T, my_sim.settings,
        my_sim.initial_conditions)
    my_sim.h_transport_problem.u = solute

    # run
    output = my_sim.make_output()

    # test
    for key in ["mesh", "solutions"]:
        assert key in output.keys()

    assert isinstance(output["mesh"], fenics.Mesh)

    for key in ["solute", "T", "retention"]:
        assert key in output["solutions"].keys()
        assert isinstance(output["solutions"][key], fenics.Function)

    retention_computed = output["solutions"]["retention"]
    solute = output["solutions"]["solute"]
    print(retention_expected(0.5, 0.5))
    print(retention_computed(0.5, 0.5))
    assert fenics.errornorm(retention_computed, retention_expected) == \
        pytest.approx(0)


def test_output_of_run_without_traps_with_chemical_pot():
    '''
    Test method make_output() returns a dict with the appropriate keys
    without traps with chemical pot
    '''
    # build
    mesh = fenics.UnitSquareMesh(10, 10)
    V_CG1 = fenics.FunctionSpace(mesh, 'CG', 1)
    V_DG1 = fenics.FunctionSpace(mesh, 'DG', 1)
    my_sim = FESTIM.Simulation(
        {"boundary_conditions": [], "exports": {}, "traps": []})
    my_sim.mesh = FESTIM.Mesh(mesh)
    my_sim.settings.chemical_pot = True
    my_sim.mobile = FESTIM.Theta()
    my_sim.mobile.S = fenics.Constant(3)
    my_temp = FESTIM.Temperature("solve_stationary")
    my_temp.T = fenics.Function(V_CG1)
    my_sim.T = my_temp
    my_sim.V_DG1 = V_DG1
    my_sim.h_transport_problem = FESTIM.HTransportProblem(
        my_sim.mobile, my_sim.traps, my_sim.T, my_sim.settings,
        my_sim.initial_conditions)
    # concentrations
    val_solute = 1

    solute = fenics.project(
        fenics.Expression(str(val_solute), degree=0)/my_sim.mobile.S,
        V_CG1)

    retention_expected = fenics.project(val_solute, V_CG1)
    my_sim.h_transport_problem.u = solute
    my_sim.exports = FESTIM.Exports([])
    # run
    output = my_sim.make_output()

    # test
    for key in ["mesh", "solutions"]:
        assert key in output.keys()

    assert isinstance(output["mesh"], fenics.Mesh)

    for key in ["solute", "T", "retention"]:
        assert key in output["solutions"].keys()
        assert isinstance(output["solutions"][key], fenics.Function)

    retention_computed = output["solutions"]["retention"]
    solute = output["solutions"]["solute"]
    print(retention_expected(0.5, 0.5))
    print(retention_computed(0.5, 0.5))
    assert fenics.errornorm(retention_computed, retention_expected) == \
        pytest.approx(0)


def test_output_of_run_with_traps_with_chemical_pot():
    '''
    Test method make_output() returns a dict with the appropriate keys
    with traps and chemical pot
    '''
    # # build
    mesh = fenics.UnitSquareMesh(10, 10)
    V_CG1 = fenics.VectorFunctionSpace(mesh, 'CG', 1, 3)
    V_DG1 = fenics.FunctionSpace(mesh, 'DG', 1)
    my_sim = FESTIM.Simulation()
    traps = [
        FESTIM.Trap(1, 1, 1, 1, 1, 1),
        FESTIM.Trap(1, 1, 1, 1, 1, 1),
    ]
    my_sim.traps = FESTIM.Traps(traps)
    my_sim.mesh = FESTIM.Mesh(mesh)
    my_sim.settings = FESTIM.Settings(1e10, 1e-10, chemical_pot=True)
    my_sim.mobile = FESTIM.Theta()
    my_sim.mobile.S = fenics.Constant(3)

    my_temp = FESTIM.Temperature("solve_stationary")
    my_temp.T = fenics.Function(V_CG1)
    my_sim.T = my_temp
    my_sim.V_DG1 = V_DG1
    my_sim.h_transport_problem = FESTIM.HTransportProblem(
        my_sim.mobile, my_sim.traps, my_sim.T, my_sim.settings,
        my_sim.initial_conditions)
    my_sim.h_transport_problem.settings = my_sim.settings
    my_sim.h_transport_problem.define_function_space(my_sim.mesh)
    my_sim.h_transport_problem.initialise_concentrations(my_sim.materials)
    my_sim.exports = FESTIM.Exports([])
    # concentrations
    val_solute = 1
    val_trap_1 = 2
    val_trap_2 = 3
    u = fenics.Function(V_CG1)

    solute = fenics.project(
        fenics.Expression(str(val_solute), degree=0)/my_sim.mobile.S,
        V_CG1.sub(0).collapse())
    trap_1 = fenics.interpolate(
        fenics.Expression(str(val_trap_1), degree=0),
        V_CG1.sub(1).collapse())
    trap_2 = fenics.interpolate(
        fenics.Expression(str(val_trap_2), degree=0),
        V_CG1.sub(2).collapse())

    retention_expected = fenics.project(
        val_solute + val_trap_1 + val_trap_2, V_DG1)

    fenics.assign(u.sub(0), solute)
    fenics.assign(u.sub(1), trap_1)
    fenics.assign(u.sub(2), trap_2)
    my_sim.h_transport_problem.u = u
    # run
    output = my_sim.make_output()

    # test
    for key in ["mesh", "solutions"]:
        assert key in output.keys()
    assert isinstance(output["mesh"], fenics.Mesh)
    for key in ["solute", "T", "trap_1", "trap_2", "retention"]:
        print(key, output["solutions"][key])
        print(type(output["solutions"][key]))
        assert key in output["solutions"].keys()
        assert isinstance(output["solutions"][key], fenics.Function)

    retention_computed = output["solutions"]["retention"]
    solute = output["solutions"]["solute"]
    trap_1 = output["solutions"]["trap_1"]
    trap_2 = output["solutions"]["trap_2"]
    print(retention_expected(0.5, 0.5))
    print(retention_computed(0.5, 0.5))
    assert fenics.errornorm(retention_computed, retention_expected) == \
        pytest.approx(0)
