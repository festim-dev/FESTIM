from FESTIM.export import write_to_csv, export_xdmf, export_parameters, \
                          treat_value
from FESTIM import x
import FESTIM
import fenics
from pathlib import Path
import pytest


def test_write_to_csv(tmpdir):
    """Tests the export.function write_to_csv()
    """
    d = tmpdir.mkdir("out")
    data = [[1, 2, 3, 4, 5]]
    derived_quantities_dict = {
        "folder": str(Path(d)),
        "file":  "out"
    }
    assert write_to_csv(derived_quantities_dict, data)

    derived_quantities_dict = {
        "file": str(Path(d)) + "/out.csv"
    }
    assert write_to_csv(derived_quantities_dict, data)


def test_export_xdmf(tmpdir):
    """Tests the several errors that can be raised by export.export_xdmf()
    """
    d = tmpdir.mkdir("out")

    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    res = [fenics.Function(V), fenics.Function(V)]
    files = [fenics.XDMFFile(str(Path(d)) + "/solute.xdmf"),
             fenics.XDMFFile(str(Path(d)) + "/1.xdmf")]

    with pytest.raises(NameError, match=r'Too many functions to export'):
        exports = {
            "xdmf": {
                "functions": ['solute', '1', '3'],
                "labels": ['solute', '1'],
                "folder": str(Path(d))
            }
        }
        export_xdmf(res, exports, files, t=1, append=False)

    with pytest.raises(TypeError, match=r'checkpoint'):
        exports = {
            "xdmf": {
                "functions": ['solute', '1'],
                "labels": ['solute', '1'],
                "checkpoint": "False",
                "folder": str(Path(d))
            }
        }
        export_xdmf(res, exports, files, t=1, append=False)

    exports["xdmf"].update({"checkpoint": False})
    export_xdmf(res, exports, files, t=1, append=False)

    with pytest.raises(ValueError, match=r'trap1'):
        exports = {
            "xdmf": {
                "functions": ['solute', 'trap1'],
                "labels": ['solute', '1'],
                "folder": str(Path(d))
            }
        }
        export_xdmf(res, exports, files, t=1, append=False)

    with pytest.raises(TypeError, match=r'type'):
        exports = {
            "xdmf": {
                "functions": ['solute', 1.2],
                "labels": ['solute', '1'],
                "folder": str(Path(d))
            }
        }
        export_xdmf(res, exports, files, t=1, append=False)

    with pytest.raises(ValueError, match=r'3'):
        exports = {
            "xdmf": {
                "functions": ['solute', 3],
                "labels": ['solute', '1'],
                "folder": str(Path(d))
            }
        }
        export_xdmf(res, exports, files, t=1, append=False)
    return


def test_export_parameters(tmpdir):
    """Tests the function export parameters
    """
    d = tmpdir.mkdir("out")

    def thermal_cond(T):
        return 2*T

    parameters = {
        "exports": {
            "parameters": str(Path(d)) + "/parameters"
        },
        "a": {
            "a1": x,
            "a2": 3
        },
        "b": [
            {
                "b11": 'a',
                "b12": 'b'
            },
            {
                "b21": 'a',
                "b22": 'b'
            },
        ],
        "thermal_cond": thermal_cond
    }
    assert export_parameters(parameters)


def test_treat_value():
    """Tests that the function export.treat_value() returns the correct dict
    """
    d = {
        "a": {
            "a1": x,
            "a2": 3
        },
        "b": [
            {
                "b11": 'a',
                "b12": ['1', 2]
            },
            {
                "b21": 'a',
                "b22": 'b'
            },
        ],
        "thermal_cond": 2
    }
    d_expected = {
        "a": {
            "a1": 'x[0]',
            "a2": 3
        },
        "b": [
            {
                "b11": 'a',
                "b12": ['1', 2]
            },
            {
                "b21": 'a',
                "b22": 'b'
            },
        ],
        "thermal_cond": 2
    }
    assert treat_value(d) == d_expected


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
    my_sim.u = fenics.Function(V_CG1)
    my_temp = FESTIM.Temperature("solve_stationary")
    my_temp.T = fenics.Function(V_CG1)
    my_sim.T = my_temp
    my_sim.V_DG1 = V_DG1

    # concentrations
    val_solute = 1

    solute = fenics.project(
        fenics.Expression(str(val_solute), degree=0),
        V_CG1)

    retention_expected = fenics.project(val_solute, V_CG1)
    my_sim.u = solute

    # run
    output = my_sim.make_output()

    # test
    for key in ["parameters", "mesh", "solutions"]:
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
    my_sim.chemical_pot = True
    my_sim.S = fenics.Constant(3)
    my_temp = FESTIM.Temperature("solve_stationary")
    my_temp.T = fenics.Function(V_CG1)
    my_sim.T = my_temp
    my_sim.V_DG1 = V_DG1

    # concentrations
    val_solute = 1

    solute = fenics.project(
        fenics.Expression(str(val_solute), degree=0)/my_sim.S,
        V_CG1)

    retention_expected = fenics.project(val_solute, V_CG1)
    my_sim.u = solute

    # run
    output = my_sim.make_output()

    # test
    for key in ["parameters", "mesh", "solutions"]:
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
    my_sim = FESTIM.Simulation(
        {"boundary_conditions": [], "exports": {}, "traps": [{}, {}]})
    my_sim.mesh = FESTIM.Mesh(mesh)
    my_sim.chemical_pot = True
    my_sim.S = fenics.Constant(3)
    my_temp = FESTIM.Temperature("solve_stationary")
    my_temp.T = fenics.Function(V_CG1)
    my_sim.T = my_temp
    my_sim.V_DG1 = V_DG1

    # concentrations
    val_solute = 1
    val_trap_1 = 2
    val_trap_2 = 3
    u = fenics.Function(V_CG1)

    solute = fenics.project(
        fenics.Expression(str(val_solute), degree=0)/my_sim.S,
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

    my_sim.u = u

    # run
    output = my_sim.make_output()

    # test
    for key in ["parameters", "mesh", "solutions"]:
        assert key in output.keys()
    assert isinstance(output["mesh"], fenics.Mesh)

    for key in ["solute", "T", "trap_1", "trap_2", "retention"]:
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
