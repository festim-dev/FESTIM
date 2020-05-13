from FESTIM.export import write_to_csv, export_xdmf, export_parameters
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
            "a1": 2,
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
