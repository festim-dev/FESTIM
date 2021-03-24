import os.path
from os import path
import FESTIM
from FESTIM.meshing import subdomains_1D
from FESTIM.post_processing import header_derived_quantities,\
    create_properties, run_post_processing
from FESTIM.export import define_xdmf_files
import fenics
import pytest
import sympy as sp
import numpy as np
from pathlib import Path
import timeit


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
                "E_D": 4,
                "D_0": 5,
                "thermal_cond": 1,
                "id": 1
                },
                {
                "borders": [0.5, 1],
                "E_D": 5,
                "D_0": 6,
                "thermal_cond": 1,
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
                        "field": 'retention',
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
    V_DG1 = fenics.FunctionSpace(mesh, 'DG', 1)
    W = fenics.FunctionSpace(mesh, 'P', 1)
    u = fenics.Function(V)
    T = fenics.interpolate(fenics.Constant(100), W)

    volume_markers, surface_markers = \
        subdomains_1D(mesh, parameters["materials"], size=1)

    t = 0
    dt = 1

    files = define_xdmf_files(parameters["exports"])
    tab = \
        [header_derived_quantities(parameters)]
    properties = \
        create_properties(
            mesh, parameters["materials"], volume_markers, T)
    my_sim = FESTIM.Simulation(parameters)
    my_sim.transient = True
    my_sim.u = u
    my_sim.T = T
    my_sim.volume_markers, my_sim.surface_markers = \
        volume_markers, surface_markers
    my_sim.V_CG1, my_sim.V_DG1 = V, V_DG1
    my_sim.dt = dt
    my_sim.files = files
    my_sim.append = False
    my_sim.D, my_sim.thermal_cond, my_sim.cp, my_sim.rho, \
        my_sim.H, my_sim.S = properties
    my_sim.derived_quantities_global = []
    for i in range(1, 3):
        t += dt
        my_sim.t = t
        my_sim.derived_quantities_global, dt = \
            run_post_processing(my_sim)
        my_sim.append = True
    assert len(my_sim.derived_quantities_global) == i + 1
    assert my_sim.derived_quantities_global[i][0] == t


def test_run_post_processing_pure_diffusion(tmpdir):
    '''
    Test the integration of post processing functions.
    In the pure diffusion case
    '''
    d = tmpdir.mkdir("Solution_Test")
    parameters = {
        "materials": [{
                "borders": [0, 0.5],
                "E_D": 4,
                "D_0": 5,
                "id": 1
                },
                {
                "borders": [0.5, 1],
                "E_D": 5,
                "D_0": 6,
                "id": 2
                }],
        "traps": [
            {}
        ],
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
                "minimum_volume": [
                    {
                        "field": "retention",
                        "volumes": [1]
                    },
                    {
                        "field": "1",
                        "volumes": [1]
                    },
                ],
                "file": "derived_quantities",
                "folder": str(Path(d)),
                },
            }
        }

    mesh = fenics.UnitIntervalMesh(20)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)
    W = fenics.FunctionSpace(mesh, 'P', 1)
    V_DG1 = fenics.FunctionSpace(mesh, 'DG', 1)
    u = fenics.Function(V)
    fenics.assign(u.sub(0), fenics.interpolate(fenics.Constant(10), V.sub(0).collapse()))
    fenics.assign(u.sub(1), fenics.interpolate(fenics.Constant(1), V.sub(1).collapse()))
    T = fenics.interpolate(fenics.Constant(20), W)

    volume_markers, surface_markers = \
        subdomains_1D(mesh, parameters["materials"], size=1)

    t = 0
    dt = 1
    files = define_xdmf_files(parameters["exports"])
    tab = \
        [header_derived_quantities(parameters)]
    properties = \
        create_properties(
            mesh, parameters["materials"], volume_markers, T)
    my_sim = FESTIM.Simulation(parameters)
    my_sim.transient = True
    my_sim.u = u
    my_sim.T = T
    my_sim.volume_markers, my_sim.surface_markers = \
        volume_markers, surface_markers
    my_sim.V_CG1, my_sim.V_DG1 = V, V_DG1
    my_sim.dt = dt
    my_sim.files = files
    my_sim.append = False
    my_sim.D, my_sim.thermal_cond, my_sim.cp, my_sim.rho, \
        my_sim.H, my_sim.S = properties
    my_sim.derived_quantities_global = []

    for i in range(1, 3):
        t += dt
        my_sim.t = t
        my_sim.derived_quantities_global, my_sim.dt = \
            run_post_processing(my_sim)
        my_sim.append = True
        assert len(my_sim.derived_quantities_global) == i + 1
        assert my_sim.derived_quantities_global[i][0] == t
        assert my_sim.derived_quantities_global[i][1] == 10
        assert my_sim.derived_quantities_global[i][2] == 20
        assert round(my_sim.derived_quantities_global[i][3]) == 11
        assert my_sim.derived_quantities_global[i][4] == 11
        assert my_sim.derived_quantities_global[i][5] == 1


def test_run_post_processing_flux(tmpdir):
    '''
    Test run_post_processing() quantitatively
    '''
    d = tmpdir.mkdir("Solution_Test")
    parameters = {
        "materials": [{
                "borders": [0, 0.5],
                "E_D": 0.4,
                "D_0": 5,
                "thermal_cond": 3,
                "id": 1
                },
                {
                "borders": [0.5, 1],
                "E_D": 0.5,
                "D_0": 6,
                "thermal_cond": 5,
                "id": 2
                }],
        "traps": [],
        "exports": {
            "derived_quantities": {
                "surface_flux": [
                    {
                        "field": 'solute',
                        "surfaces": [1, 2]
                    },
                    {
                        "field": 'T',
                        "surfaces": [1, 2]
                    }
                    ],
                "file": "derived_quantities",
                "folder": str(Path(d)),
                },
            }
        }

    mesh = fenics.UnitIntervalMesh(20)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    V_DG1 = fenics.FunctionSpace(mesh, 'DG', 1)
    u = fenics.Expression('2*x[0]', degree=1)
    u = fenics.interpolate(u, V)
    T = fenics.Expression('100*x[0] + 200', degree=1)
    T = fenics.interpolate(T, V)

    volume_markers, surface_markers = \
        subdomains_1D(mesh, parameters["materials"], size=1)

    t = 0
    dt = 1
    transient = True
    append = True
    markers = [volume_markers, surface_markers]

    # files = define_xdmf_files(parameters["exports"])
    tab = [header_derived_quantities(parameters)]
    properties = \
        create_properties(
            mesh, parameters["materials"], volume_markers, T)
    t += dt
    my_sim = FESTIM.Simulation(parameters)
    my_sim.transient = True
    my_sim.u = u
    my_sim.T = T
    my_sim.volume_markers, my_sim.surface_markers = volume_markers, surface_markers
    my_sim.V_CG1, my_sim.V_DG1 = V, V_DG1
    my_sim.dt = dt
    my_sim.t = t
    my_sim.files = None
    my_sim.append = False
    my_sim.D, my_sim.thermal_cond, my_sim.cp, my_sim.rho, \
        my_sim.H, my_sim.S = properties
    my_sim.derived_quantities_global = []
    derived_quantities_global, dt = \
        run_post_processing(my_sim)
    print(derived_quantities_global[0])
    print(derived_quantities_global[1])
    assert np.isclose(derived_quantities_global[1][1],
                      -1*2*5*fenics.exp(-0.4/FESTIM.k_B/T(0)))
    assert np.isclose(derived_quantities_global[1][2],
                      -1*-2*6*fenics.exp(-0.5/FESTIM.k_B/T(1)))
    assert np.isclose(derived_quantities_global[1][3], -1*100*3)
    assert np.isclose(derived_quantities_global[1][4], -1*-100*5)


def test_performance_xdmf_export_every_N_iterations(tmpdir):
    '''
    Test the postprocessing runs faster when exporting to XDMF less often
    '''
    d = tmpdir.mkdir("Solution_Test")
    parameters = {
        "materials": [{
                "borders": [0, 0.5],
                "E_D": 1,
                "D_0": 1,
                "id": 1
                }],
        "traps": [
            {}
        ],
        "exports": {
            "xdmf": {
                    "functions": ['solute', 'T'],
                    "labels":  ['solute', 'temperature'],
                    "folder": str(Path(d)),
                    },

            }
        }
    mesh = fenics.UnitSquareMesh(16, 16)
    V_CG1 = fenics.FunctionSpace(mesh, 'CG', 1)
    V_DG1 = fenics.FunctionSpace(mesh, 'DG', 1)

    my_sim = FESTIM.Simulation(parameters)
    my_sim.T = fenics.Function(V_CG1)
    my_sim.u = fenics.Function(V_CG1)
    my_sim.V_CG1 = V_CG1
    my_sim.V_DG1 = V_DG1
    my_sim.volume_markers, my_sim.surface_markers = 'foo', 'foo'
    my_sim.t = 0
    my_sim.append = False
    my_sim.D, my_sim.thermal_cond, my_sim.cp, my_sim.rho, \
        my_sim.H, my_sim.S = 0, 0, 0, 0, 0, None,
    my_sim.files = FESTIM.define_xdmf_files(parameters["exports"])

    # export every time
    my_sim.nb_iterations_between_exports = 1
    start = timeit.default_timer()
    for i in range(30):
        my_sim.nb_iterations += 1
        FESTIM.run_post_processing(my_sim)

    stop = timeit.default_timer()
    long_time = stop - start
    print(long_time)

    # export every 10 iterations
    my_sim.nb_iterations = 0
    my_sim.append = False
    my_sim.nb_iterations_between_exports = 20
    start = timeit.default_timer()
    for i in range(30):
        my_sim.nb_iterations += 1
        FESTIM.run_post_processing(my_sim)

    stop = timeit.default_timer()
    short_time = stop - start
    print(short_time)

    assert short_time < long_time


def test_performance_xdmf_export_only_last_timestep(tmpdir):
    '''
    Test that the XDMF export isn't done until the last timestep
    '''
    d = tmpdir.mkdir("Solution_Test")
    parameters = {
        "materials": [{
                "borders": [0, 0.5],
                "E_D": 1,
                "D_0": 1,
                "id": 1
                }],
        "traps": [
            {}
        ],
        "exports": {
            "xdmf": {
                    "functions": ['solute', 'T'],
                    "labels":  ['solute', 'temperature'],
                    "folder": str(Path(d)),
                    },
            }
        }
    mesh = fenics.UnitSquareMesh(16, 16)
    V_CG1 = fenics.FunctionSpace(mesh, 'CG', 1)
    V_DG1 = fenics.FunctionSpace(mesh, 'DG', 1)

    my_sim = FESTIM.Simulation(parameters)
    my_sim.T = fenics.Function(V_CG1)
    my_sim.u = fenics.Function(V_CG1)
    my_sim.V_CG1 = V_CG1
    my_sim.V_DG1 = V_DG1
    my_sim.volume_markers, my_sim.surface_markers = 'foo', 'foo'
    my_sim.t = 0
    my_sim.final_time = 100
    my_sim.append = False
    my_sim.D, my_sim.thermal_cond, my_sim.cp, my_sim.rho, \
        my_sim.H, my_sim.S = 0, 0, 0, 0, 0, None,
    my_sim.files = FESTIM.define_xdmf_files(parameters["exports"])
    my_sim.export_xdmf_last_only = True

    run_post_processing(my_sim)
    assert not path.exists(str(Path(d)) + '/solute.xdmf')

    my_sim.t = my_sim.final_time
    run_post_processing(my_sim)
    assert path.exists(str(Path(d)) + '/solute.xdmf')


def test_derived_quantities_global():

    parameters = {
        "mesh_parameters": {
            "initial_number_of_cells": 10,
            "size": 1,
        },
        "materials": [
            {
                "D_0": 1,
                "E_D": 1,
                "id": 1,
            },
            ],
        "traps": [
            ],
        "boundary_conditions": [
            ],
        "temperature": {
            "type": "expression",
            "value": 300
            },
        "solving_parameters": {
            "final_time": 1,
            "initial_stepsize": 0.01,
          },
        "exports": {
            "derived_quantities": {
                "surface_flux": [
                    {
                        "surfaces": [1, 2],
                        "field": "solute"
                    }
                ],
                "file": "derived_quantities.csv",
                "folder": 'out'
            }
        }

    }
    my_sim = FESTIM.Simulation(parameters)
    my_sim.initialise()
    my_sim.t = 0
    for i in range(10):
        run_post_processing(my_sim)
        assert len(my_sim.derived_quantities_global) == i + 2
        my_sim.append = True
