import FESTIM
import fenics
import pytest
import sympy as sp
from pathlib import Path


def test_run_post_processing(tmpdir):
    '''
    Test the integration of post processing functions.
    Check the derived quantities table sizes, and the value
    of t
    '''
    d = tmpdir.mkdir("Solution_Test")
    parameters = {
        "materials": [{
                "borders": [0, 0.5],
                "E_diff": 4,
                "D_0": 5,
                "id": 1
                },
                {
                "borders": [0.5, 1],
                "E_diff": 5,
                "D_0": 6,
                "id": 2
                }],
        "traps": [{}, {}],
        "exports": {
            "xdmf": {
                    "functions": ['solute', 'T'],
                    "labels":  ['solute', 'temperature'],
                    "folder": str(Path(d))
            },
            "derived_quantities": {
                "surface_flux": [
                    {
                        "field": 'solute',
                        "surfaces": [2]
                    },
                    {
                        "field": 'T',
                        "surfaces": [2]
                    },
                ],
                "average_volume": [
                    {
                        "field": 1,
                        "volumes": [1]
                    }
                ],
                "total_volume": [
                    {
                        "field": 'solute',
                        "volumes": [1, 2]
                    }
                ],
                "total_surface": [
                    {
                        "field": 1,
                        "surfaces": [2]
                    }
                ],
                "maximum_volume": [
                    {
                        "field": 'T',
                        "volumes": [1]
                    }
                ],
                "minimum_volume": [
                    {
                        "field": 'solute',
                        "volumes": [2]
                    }
                ],
                "file": "derived_quantities",
                "folder": str(Path(d)),
                },
                }
    }

    mesh = fenics.UnitIntervalMesh(20)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)
    W = fenics.FunctionSpace(mesh, 'P', 1)
    u = fenics.Function(V)
    T = fenics.Function(W)

    volume_markers, surface_markers = \
        FESTIM.meshing.subdomains_1D(mesh, parameters["materials"], size=1)

    t = 0
    dt = 1
    transient = True
    append = True
    markers = [volume_markers, surface_markers]

    files = FESTIM.export.define_xdmf_files(parameters["exports"])
    tab = \
        [FESTIM.post_processing.header_derived_quantities(parameters)]
    flux_fonctions = \
        FESTIM.post_processing.create_flux_functions(
            mesh, parameters["materials"], volume_markers)
    for i in range(1, 3):
        t += dt
        derived_quantities_global, dt = \
            FESTIM.post_processing.run_post_processing(
                parameters, transient, u, T, markers, W, t, dt, files,
                append=append, flux_fonctions=flux_fonctions,
                derived_quantities_global=tab)
        append = True
    assert len(derived_quantities_global) == i + 1
    assert derived_quantities_global[i][0] == t


def test_run_post_processing_pure_diffusion(tmpdir):
    '''
    Test the integration of post processing functions.
    In the pure diffusion case
    '''
    d = tmpdir.mkdir("Solution_Test")
    parameters = {
        "materials": [{
                "borders": [0, 0.5],
                "E_diff": 4,
                "D_0": 5,
                "id": 1
                },
                {
                "borders": [0.5, 1],
                "E_diff": 5,
                "D_0": 6,
                "id": 2
                }],
        "traps": [],
        "exports": {
            "xdmf": {
                    "functions": ['solute', 'T'],
                    "labels":  ['solute', 'temperature'],
                    "folder": str(Path(d))
                    },
            "derived_quantities": {
                "average_volume": [
                    {
                        "field": 'solute',
                        "volumes": [2]
                    },
                    {
                        "field": 'T',
                        "volumes": [2]
                    },
                    {
                        "field": 'retention',
                        "volumes": [2]
                    },
                    ],
                "file": "derived_quantities",
                "folder": str(Path(d)),
                },
            }
        }

    mesh = fenics.UnitIntervalMesh(20)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    u = fenics.interpolate(fenics.Constant(10), V)
    T = fenics.interpolate(fenics.Constant(20), V)

    volume_markers, surface_markers = \
        FESTIM.meshing.subdomains_1D(mesh, parameters["materials"], size=1)

    t = 0
    dt = 1
    transient = True
    append = True
    markers = [volume_markers, surface_markers]

    files = FESTIM.export.define_xdmf_files(parameters["exports"])
    tab = \
        [FESTIM.post_processing.header_derived_quantities(parameters)]
    flux_fonctions = \
        FESTIM.post_processing.create_flux_functions(
            mesh, parameters["materials"], volume_markers)
    for i in range(1, 3):
        t += dt
        derived_quantities_global, dt = \
            FESTIM.post_processing.run_post_processing(
                parameters, transient, u, T, markers, V, t, dt, files,
                append=append, flux_fonctions=flux_fonctions,
                derived_quantities_global=tab)
        append = True
        assert len(derived_quantities_global) == i + 1
        assert derived_quantities_global[i][0] == t
        assert derived_quantities_global[i][1] == 10
        assert derived_quantities_global[i][2] == 20
        assert round(derived_quantities_global[i][3]) == 10
