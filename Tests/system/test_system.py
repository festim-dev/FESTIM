import FESTIM
from FESTIM.generic_simulation import run
import fenics
import pytest
import sympy as sp
import numpy as np
from pathlib import Path
import timeit


# System tests

def test_run_temperature_stationary(tmpdir):
    '''
    Check that the temperature module works well in 1D stationary
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + 2*FESTIM.x**2
    size = 1
    parameters = {
        "materials": [
            {
                "thermal_cond": 1,
                # "borders": [0, size],
                "E_D": 0.39,
                "D_0": 4.1e-7,
                "id": 1,
                }
                ],
        "traps": [
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 200,
                "size": size,
                "refinements": [
                ],
            },
        "boundary_conditions": [
                    {
                        "surfaces": 1,
                        "value": 1,
                        "component": 0,
                        "type": "dc"
                    }
            ],
        "temperature": {
            "type": "solve_stationary",
            "boundary_conditions": [
                {
                    "type": "dc",
                    "value": u,
                    "surfaces": [1, 2]
                }
                ],
            "source_term": [
                {
                    "value": -4,
                    "volume": 1
                }
            ],
        },
        "source_term": {
            'value': 0
            },
        "solving_parameters": {
            "final_time": 30,
            "initial_stepsize": 0.5,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1,
                "dt_min": 1e-5
                },
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 50,
            }
            },
        "exports": {
            "parameters":  str(Path(d)) + "/param.json",
            "xdmf": {
                    "fields": ['T', 'solute'],
                    "labels":  ['temperature', 'solute'],
                    "folder": str(Path(d))
            },
            "derived_quantities": {
                "total_volume": [
                    {
                        "volumes": [1],
                        "field": "solute"
                    },
                ],
                "file": "derived_quantities.csv",
                "folder": str(Path(d))
            },
            "error": [
                {
                    "computed_solutions": ['T'],
                    "exact_solutions": [u],
                    "norm": 'error_max',
                    "degree": 4
                }
            ]
            },

    }
    output = run(parameters)
    assert output["error"][0][1] < 1e-9


def test_run_temperature_transient(tmpdir):
    '''
    Check that the temperature module works well in 1D transient
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + 2*FESTIM.x**2+FESTIM.t
    size = 1
    parameters = {
        "materials": [
            {
                "thermal_cond": 1,
                "rho": 1,
                "heat_capacity": 1,
                "borders": [0, size],
                "E_D": 0.39,
                "D_0": 4.1e-7,
                "id": 1
            }
            ],
        "traps": [
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 200,
                "size": size,
                "refinements": [
                ],
            },
        "boundary_conditions": [
                    {
                        "surfaces": [1],
                        "value": 1,
                        "component": 0,
                        "type": "dc"
                    }
            ],
        "temperature": {
            "type": "solve_transient",
            "boundary_conditions": [
                {
                    "type": "dc",
                    "value": u,
                    "surfaces": 1
                },
                {
                    "type": "dc",
                    "value": u,
                    "surfaces": 2
                },
                ],
            "source_term": [
                {
                    "value": sp.diff(u, FESTIM.t) - sp.diff(u, FESTIM.x, 2),
                    "volume": 1
                }
            ],
            "initial_condition": u
        },
        "source_term": {
            'value': 0
            },
        "solving_parameters": {
            "type": "solve_transient",
            "final_time": 30,
            "initial_stepsize": 0.5,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1,
                "t_stop": 40,
                "stepsize_stop_max": 0.5,
                "dt_min": 1e-5
                },
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 50,
            }
            },
        "exports": {
            "xdmf": {
                    "fields": ['T'],
                    "labels":  ['temperature'],
                    "folder": str(Path(d))
            },
            "error": [
                {
                    "computed_solutions": ['T'],
                    "exact_solutions": [u],
                    "norm": 'error_max',
                    "degree": 4
                }
            ]
            },

    }
    output = run(parameters)

    assert output["error"][0][1] < 1e-9


def test_run_MMS(tmpdir):
    '''
    Test function run() for several refinements
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + sp.sin(2*fenics.pi*FESTIM.x)*FESTIM.t
    v = 1 + sp.cos(2*fenics.pi*FESTIM.x)*FESTIM.t

    def parameters(h, dt, final_time, u, v):
        size = 1
        k_0 = 2
        E_k = 1.5
        p_0 = 3
        E_p = 0.2
        T = 700 + 30*FESTIM.x
        n_trap = 1
        E_D = 0.1
        D_0 = 2
        k_B = FESTIM.k_B
        D = D_0 * sp.exp(-E_D/k_B/T)
        p = p_0 * sp.exp(-E_p/k_B/T)
        k = k_0 * sp.exp(-E_k/k_B/T)

        f = sp.diff(u, FESTIM.t) + sp.diff(v, FESTIM.t) - \
            D * sp.diff(u, FESTIM.x, 2) - \
            sp.diff(D, FESTIM.x)*sp.diff(u, FESTIM.x)
        g = sp.diff(v, FESTIM.t) + p*v - k * u * (n_trap-v)
        parameters = {
            "materials": [
                {
                    "borders": [0, size],
                    "E_D": E_D,
                    "D_0": D_0,
                    "id": 1
                    }
                    ],
            "traps": [
                {
                    "E_k": E_k,
                    "k_0": k_0,
                    "E_p": E_p,
                    "p_0": p_0,
                    "density": n_trap,
                    "materials": 1,
                    "source_term": g
                }
                ],
            "initial_conditions": [
                {
                    "value": u,
                    "component": 0
                },
                {
                    "value": v,
                    "component": 1
                }
            ],

            "mesh_parameters": {
                    "initial_number_of_cells": round(size/h),
                    "size": size,
                    "refinements": [
                    ],
                },
            "boundary_conditions": [
                    {
                        "surfaces": [1, 2],
                        "value": u,
                        "component": 0,
                        "type": "dc"
                    },
                    {
                        "surfaces": [1, 2],
                        "value": v,
                        "component": 1,
                        "type": "dc"
                    }
                ],
            "temperature": {
                    'type': "expression",
                    'value': T
                },
            "source_term": {
                'value': f
                },
            "solving_parameters": {
                "final_time": final_time,
                "initial_stepsize": dt,
                "adaptive_stepsize": {
                    "stepsize_change_ratio": 1,
                    "t_stop": 0,
                    "stepsize_stop_max": dt,
                    "dt_min": 1e-5
                    },
                "newton_solver": {
                    "absolute_tolerance": 1e-10,
                    "relative_tolerance": 1e-9,
                    "maximum_iterations": 50,
                }
                },
            "exports": {
                "xdmf": {
                    "fields": ['retention'],
                    "labels": ['retention'],
                    "folder": str(Path(d))
                },
                "error": [
                    {
                        "computed_solutions": [0, 1],
                        "exact_solutions": [u, v],
                        "norm": 'error_max',
                        "degree": 4
                    }
                ]
                },
        }
        return parameters

    tol_u = 1e-7
    tol_v = 1e-6
    sizes = [1/1600, 1/1700]
    dt = 0.1/50
    final_time = 0.1
    for h in sizes:
        output = run(
            parameters(h, dt, final_time, u, v))
        error_max_u = output["error"][0][1]
        error_max_v = output["error"][0][2]
        msg = 'Maximum error on u is:' + str(error_max_u) + '\n \
            Maximum error on v is:' + str(error_max_v) + '\n \
            with h = ' + str(h) + '\n \
            with dt = ' + str(dt)
        print(msg)
        assert error_max_u < tol_u and error_max_v < tol_v


def test_run_MMS_chemical_pot(tmpdir):
    '''
    Test function run() with conservation of chemical potential (1 material)
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + sp.sin(2*fenics.pi*FESTIM.x)*FESTIM.t + FESTIM.t
    v = 1 + sp.cos(2*fenics.pi*FESTIM.x)*FESTIM.t

    def parameters(h, dt, final_time, u, v):
        size = 1
        k_0 = 2
        E_k = 1.5
        p_0 = 3
        E_p = 0.2
        T = 700 + 30*FESTIM.x
        n_trap = 1
        E_D = 0.1
        D_0 = 2
        k_B = FESTIM.k_B
        D = D_0 * sp.exp(-E_D/k_B/T)
        p = p_0 * sp.exp(-E_p/k_B/T)
        k = k_0 * sp.exp(-E_k/k_B/T)

        f = sp.diff(u, FESTIM.t) + sp.diff(v, FESTIM.t) - \
            D * sp.diff(u, FESTIM.x, 2) - \
            sp.diff(D, FESTIM.x)*sp.diff(u, FESTIM.x)
        g = sp.diff(v, FESTIM.t) + p*v - k * u * (n_trap-v)
        parameters = {
            "materials": [
                {
                    "borders": [0, size],
                    "S_0": 2,
                    "E_S": 0.1,
                    "E_D": E_D,
                    "D_0": D_0,
                    "id": 1
                    }
                    ],
            "traps": [
                {
                    "E_k": E_k,
                    "k_0": k_0,
                    "E_p": E_p,
                    "p_0": p_0,
                    "density": n_trap,
                    "materials": 1,
                    "source_term": g
                }
                ],
            "initial_conditions": [
                {
                    "value": u,
                    "component": 0
                },
                {
                    "value": v,
                    "component": 1
                }
            ],

            "mesh_parameters": {
                    "initial_number_of_cells": round(size/h),
                    "size": size,
                    "refinements": [
                    ],
                },
            "boundary_conditions": [
                    {
                        "surfaces": [1, 2],
                        "value": u,
                        "component": 0,
                        "type": "dc"
                    },
                    {
                        "surfaces": [1, 2],
                        "value": v,
                        "component": 1,
                        "type": "dc"
                    }
                ],
            "temperature": {
                    'type': "expression",
                    'value': T
                },
            "source_term": {
                'value': f
                },
            "solving_parameters": {
                "final_time": final_time,
                "initial_stepsize": dt,
                "adaptive_stepsize": {
                    "stepsize_change_ratio": 1,
                    "t_stop": 0,
                    "stepsize_stop_max": dt,
                    "dt_min": 1e-5
                    },
                "newton_solver": {
                    "absolute_tolerance": 1e-10,
                    "relative_tolerance": 1e-9,
                    "maximum_iterations": 50,
                }
                },
            "exports": {
                "txt": {
                    "functions": ["solute"],
                    "times": [100],
                    "labels": ["solute"],
                    "folder": str(Path(d))
                },
                "error": [
                    {
                        "computed_solutions": [0, 1],
                        "exact_solutions": [u, v],
                        "norm": 'error_max',
                        "degree": 4
                    }
                ],
                },
        }
        return parameters

    tol_u = 1e-7
    tol_v = 1e-6
    sizes = [1/1600]
    dt = 0.1/50
    final_time = 0.1
    for h in sizes:
        output = run(
            parameters(h, dt, final_time, u, v))
        error_max_u = output["error"][0][1]
        error_max_v = output["error"][0][2]
        msg = 'Maximum error on u is:' + str(error_max_u) + '\n \
            Maximum error on v is:' + str(error_max_v) + '\n \
            with h = ' + str(h) + '\n \
            with dt = ' + str(dt)
        print(msg)
        assert error_max_u < tol_u and error_max_v < tol_v


def test_run_chemical_pot_mass_balance(tmpdir):
    '''
    Test that when applying conservation of chemical potential
    the mass balance is ensured
    '''
    d = tmpdir.mkdir("Solution_Test")
    size = 1
    parameters = {
        "materials": [
            {
                "borders": [0, size],
                "S_0": 2,
                "E_S": 0.1,
                "E_D": 0.1,
                "D_0": 1,
                "id": 1
                }
                ],
        "traps": [
            ],
        "initial_conditions": [
            {
                "value": 1,
                "component": 0
            }
        ],

        "mesh_parameters": {
                "initial_number_of_cells": 5,
                "size": 1,
                "refinements": [
                ],
            },
        "boundary_conditions": [
            ],
        "temperature": {
                'type': "expression",
                'value': 700 + 210*FESTIM.t
            },
        "solving_parameters": {
            "final_time": 100,
            "initial_stepsize": 2,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1,
                "t_stop": 0,
                "stepsize_stop_max": 100,
                "dt_min": 1e-5
                },
            "newton_solver": {
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 50,
            }
            },
        "exports": {
            "xdmf": {
                "fields": ["retention"],
                "labels": ["retention"],
                "folder": str(Path(d))
            },
            "derived_quantities": {
                "file": "derived_quantities.csv",
                "folder": str(Path(d)),
                "total_volume": [
                    {
                        "field": "solute",
                        "volumes": [1]
                    },
                    {
                        "field": "retention",
                        "volumes": [1]
                    },
                    ],
            },
            },
    }

    output = run(parameters)
    derived_quantities = output["derived_quantities"]
    derived_quantities.pop(0)  # remove header
    tolerance = 1e-2
    for e in derived_quantities:
        assert abs(float(e[1])-1) < tolerance
        assert abs(float(e[2])-1) < tolerance


def test_run_MMS_soret(tmpdir):
    '''
    Test function run() for several refinements with Soret effect
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + FESTIM.x**2 + FESTIM.t

    def parameters(h, dt, final_time, u):
        size = 0.1
        T = 2 + sp.cos(2*fenics.pi*FESTIM.x)*sp.cos(FESTIM.t)
        E_D = 0
        D_0 = 2
        k_B = FESTIM.k_B
        D = D_0 * sp.exp(-E_D/k_B/T)
        H = -2
        S = 3
        R = FESTIM.R
        f = sp.diff(u, FESTIM.t) - \
            sp.diff(
                (D*(sp.diff(u, FESTIM.x) +
                    (H*T+S)*u/(R*T**2)*sp.diff(T, FESTIM.x))),
                FESTIM.x)

        parameters = {
            "materials": [
                {
                    "borders": [0, size],
                    "E_D": E_D,
                    "H": {
                        "free_enthalpy": H,
                        "entropy": S
                    },
                    "D_0": D_0,
                    "id": 1
                    }
                    ],
            "traps": [],
            "initial_conditions": [
                {
                    "value": u,
                    "component": 0
                },
            ],

            "mesh_parameters": {
                    "initial_number_of_cells": round(size/h),
                    "size": size,
                    "refinements": [
                    ],
                },
            "boundary_conditions": [
                    {
                       "surfaces": [1, 2],
                       "value": u,
                       "component": 0,
                       "type": "dc"
                    }
                ],
            "temperature": {
                    'type': "expression",
                    'value': T,
                    "soret": True
                },
            "source_term": {
                'value': f
                },
            "solving_parameters": {
                "final_time": final_time,
                "initial_stepsize": dt,
                "adaptive_stepsize": {
                    "stepsize_change_ratio": 1,
                    "t_stop": 0,
                    "stepsize_stop_max": dt,
                    "dt_min": 1e-5
                    },
                "newton_solver": {
                    "absolute_tolerance": 1e-10,
                    "relative_tolerance": 1e-9,
                    "maximum_iterations": 50,
                }
                },
            "exports": {
                "txt": {
                    "fields": [],
                    "times": [],
                    "labels": [],
                    "folder": str(Path(d))
                },
                "xdmf": {
                    "fields": [0, 'T'],
                    "labels": ["solute", 'T'],
                    "folder": str(Path(d))
                },
                "error": [
                    {
                        "computed_solutions": [0],
                        "exact_solutions": [u],
                        "norm": 'L2',
                        "degree": 4
                    }
                ]
                },
        }
        return parameters

    tol_u = 1e-7
    sizes = [1/1000, 1/2000]
    dt = 0.1/50
    final_time = 0.1
    for h in sizes:
        output = run(
            parameters(h, dt, final_time, u))
        error_max_u = output["error"][0][1]
        msg = 'L2 error on u is:' + str(error_max_u) + '\n \
            with h = ' + str(h) + '\n \
            with dt = ' + str(dt)
        print(msg)
        assert error_max_u < tol_u

    return


def test_run_MMS_steady_state(tmpdir):
    '''
    Test function run() for several refinements in steady state
    '''
    d = tmpdir.mkdir("Solution_Test")
    u = 1 + FESTIM.x
    v = 1 + FESTIM.x*2

    def parameters(h, u, v):
        size = 1
        k_0 = 2
        E_k = 1.5
        p_0 = 0.2
        E_p = 0.1
        T = 700 + 30*FESTIM.x
        n_trap = 1
        E_D = 0.1
        D_0 = 2
        k_B = FESTIM.k_B
        D = D_0 * sp.exp(-E_D/k_B/T)
        p = p_0 * sp.exp(-E_p/k_B/T)
        k = k_0 * sp.exp(-E_k/k_B/T)

        f = sp.diff(u, FESTIM.t) + sp.diff(v, FESTIM.t) - \
            D * sp.diff(u, FESTIM.x, 2) - \
            sp.diff(D, FESTIM.x)*sp.diff(u, FESTIM.x)
        g = sp.diff(v, FESTIM.t) + p*v - k * u * (n_trap-v)
        parameters = {
            "materials": [
                {
                    "borders": [0, size],
                    "E_D": E_D,
                    "D_0": D_0,
                    "id": 1
                    }
                    ],
            "traps": [
                {
                    "E_k": E_k,
                    "k_0": k_0,
                    "E_p": E_p,
                    "p_0": p_0,
                    "density": n_trap,
                    "materials": 1,
                    "source_term": g
                }
                ],
            "mesh_parameters": {
                    "initial_number_of_cells": round(size/h),
                    "size": size,
                    "refinements": [
                    ],
                },
            "boundary_conditions": [
                    {
                        "surfaces": [1, 2],
                        "value": u,
                        "component": 0,
                        "type": "dc"
                    },
                    {
                        "surfaces": [1, 2],
                        "value": v,
                        "component": 1,
                        "type": "dc"
                    }
                ],
            "temperature": {
                    'type': "expression",
                    'value': T
                },
            "source_term": {
                'value': f
                },
            "solving_parameters": {
                "newton_solver": {
                    "absolute_tolerance": 1e-10,
                    "relative_tolerance": 1e-9,
                    "maximum_iterations": 50,
                },
                "type": "solve_stationary",
                "traps_element_type": 'DG'
                },
            "exports": {
                "txt": {
                    "fields": [],
                    "times": [],
                    "labels": [],
                    "folder": str(Path(d))
                },
                "xdmf": {
                    "fields": ['solute', '1', 'T', 'retention'],
                    "labels":  ['solute', '1', 'temp', 'retention'],
                    "folder": str(Path(d))
                },
                "error": [
                    {
                        "computed_solutions": [0, 1],
                        "exact_solutions": [u, v],
                        "norm": 'error_max',
                        "degree": 4
                    }
                ]
                },
        }
        return parameters

    tol_u = 1e-10
    tol_v = 1e-7
    sizes = [1/1600, 1/1700]
    for h in sizes:
        output = run(
            parameters(h, u, v))
        error_max_u = output["error"][0][1]
        error_max_v = output["error"][0][2]
        msg = 'Maximum error on u is:' + str(error_max_u) + '\n \
            Maximum error on v is:' + str(error_max_v) + '\n \
            with h = ' + str(h)
        print(msg)
        assert error_max_u < tol_u and error_max_v < tol_v


def test_chemical_pot_T_solve_stationary():
    """checks that the chemical potential conservation is well computed with
    type solve_stationary for temperature

    adapted to catch bug described in issue #310
    """
    parameters = {
        "mesh_parameters": {
            "size": 1,
            "initial_number_of_cells": 10,
            "refinements": [
            ]

            },
        "materials": [
            {
                "D_0": 1,
                "E_D": 0.1,
                "S_0": 2,
                "E_S": 0.2,
                "thermal_cond": 1,
                "id": 1,
            },
            ],
        "traps": [
            ],
        "boundary_conditions": [
            {
                "type": "dc",
                "surfaces": [1, 2],
                "value": 1
            },
            ],
        "temperature": {
            "type": "solve_stationary",
            "boundary_conditions": [
                {
                    "type": "dc",
                    "value": 300,
                    "surfaces": 1
                },
                {
                    "type": "dc",
                    "value": 300,
                    "surfaces": 2
                }
                ],
            "source_term": [
            ],
            },
        "solving_parameters": {
            "final_time": 100,
            "initial_stepsize": 10,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1.2,
                "t_stop": 1e8,
                "stepsize_stop_max": 1e7,
                "dt_min": 1e-8,
                },
            "newton_solver": {
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 20,
            }
            },
        "exports": {
            "derived_quantities": {
                "total_surface": [
                    {
                        "field": "solute",
                        "surfaces": [2]
                    }
                ]
            },
            "xdmf": {
                "fields": ['solute'],
                "labels": ["solute"],
                "folder": 'results',
            },
        }
    }
    out = run(parameters)
    assert out["derived_quantities"][-1][1] == pytest.approx(1)


def test_performance_xdmf(tmpdir):
    '''
    Check that the computation time when exporting every 10 iterations to XDMF
    is reduced
    '''
    d = tmpdir.mkdir("Solution_Test")
    parameters = {
        "materials": [
            {
                "E_D": 1,
                "D_0": 1,
                "id": 1
            }
            ],
        "traps": [
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 200,
                "size": 1,
                "refinements": [
                ],
            },
        "boundary_conditions": [
            ],
        "temperature": {
            "type": "expression",
            "value": 300
        },
        "solving_parameters": {
            "type": "solve_transient",
            "final_time": 30,
            "initial_stepsize": 4,
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 50,
            }
            },
        "exports": {
            "xdmf": {
                    "fields": ['retention', 'T'],
                    "labels":  ['retention', 'temperature'],
                    "folder": str(Path(d))
            },
            },
    }

    # long simulation
    start = timeit.default_timer()
    output = run(parameters)
    stop = timeit.default_timer()
    long_time = stop - start

    # short simulation
    parameters["exports"]["xdmf"]["nb_iterations_between_exports"] = 10
    start = timeit.default_timer()
    output = run(parameters)
    stop = timeit.default_timer()
    short_time = stop - start
    assert short_time < long_time


def test_performance_xdmf_last_timestep(tmpdir):
    '''
    Check that the computation time when exporting only the last timestep to
    XDMF is reduced
    '''
    d = tmpdir.mkdir("Solution_Test")
    parameters = {
        "materials": [
            {
                "E_D": 1,
                "D_0": 1,
                "id": 1
            }
            ],
        "traps": [
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 200,
                "size": 1,
                "refinements": [
                ],
            },
        "boundary_conditions": [
            ],
        "temperature": {
            "type": "expression",
            "value": 300
        },
        "solving_parameters": {
            "type": "solve_transient",
            "final_time": 30,
            "initial_stepsize": 3,
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 50,
            }
            },
        "exports": {
            "xdmf": {
                    "fields": ['retention', 'T'],
                    "labels":  ['retention', 'temperature'],
                    "folder": str(Path(d))
            },
            },
    }

    # long simulation
    start = timeit.default_timer()
    output = run(parameters)
    stop = timeit.default_timer()
    long_time = stop - start

    # short simulation
    parameters["exports"]["xdmf"]["last_timestep_only"] = True
    start = timeit.default_timer()
    output = run(parameters)
    stop = timeit.default_timer()
    short_time = stop - start
    assert short_time < long_time


def test_export_particle_flux_with_chemical_pot(tmpdir):
    """Checks that surface particle fluxes can be computed with conservation
    of chemical potential
    """
    d = tmpdir.mkdir("Solution_Test")
    parameters = {
        "materials": [
            {
                "E_D": 1,
                "D_0": 2,
                "E_S": 1,
                "S_0": 2,
                "thermal_cond": 2,
                "id": 1
            }
            ],
        "traps": [
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 10,
                "size": 1,
            },
        "boundary_conditions": [
            ],
        "temperature": {
            "type": "expression",
            "value": 300
        },
        "solving_parameters": {
            "type": "solve_stationary",
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 50,
            }
            },
        "exports": {
            "derived_quantities": {
                "surface_flux": [
                    {
                        "field": "solute",
                        "surfaces": [0],
                    },
                    {
                        "field": "T",
                        "surfaces": [0],
                    }
                ],
                "total_volume": [
                    {
                        "field": "retention",
                        "volumes": [1],
                    },
                ],
                "folder": str(Path(d)),
            }
            },
    }
    my_sim = FESTIM.Simulation(parameters)
    my_sim.initialise()
    my_sim.run()


def test_extrinsic_trap(tmpdir):
    """Runs a FESTIM sim with an extrinsic trap
    """

    d = tmpdir.mkdir("Solution_Test")
    parameters = {
        "materials": [
            {
                "E_D": 1,
                "D_0": 2,
                "id": 1
            }
            ],
        "traps": [
            {
                "k_0": 4.1e-7/(1.1e-10**2*6*6.3e28),
                "E_k": 0.39,
                "p_0": 1e13,
                "E_p": 1.5,
                "materials": [1],
                "type": 'extrinsic',
                "form_parameters":{
                    "phi_0": 2.5e19,
                    "n_amax": 1e-1*6.3e28,
                    "f_a": 1,
                    "eta_a": 6e-4,
                    "n_bmax": 1e-2*6.3e28,
                    "f_b": 2,
                    "eta_b": 2e-4,
                }
            }
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 10,
                "size": 1,
            },
        "boundary_conditions": [
            ],
        "temperature": {
            "type": "expression",
            "value": 300
        },
        "solving_parameters": {
            "final_time": 1,
            "initial_stepsize": 0.5,
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 10,
            }
            },
        "exports": {
            "derived_quantities": {
                "surface_flux": [
                    {
                        "field": "solute",
                        "surfaces": [0],
                    },
                ],
                "total_volume": [
                    {
                        "field": "retention",
                        "volumes": [1],
                    },
                    {
                        "field": 1,
                        "volumes": [1],
                    },
                ],
                "folder": str(Path(d)),
            }
            },
    }
    my_sim = FESTIM.Simulation(parameters)
    my_sim.initialise()
    my_sim.run()


def test_steady_state_with_2_materials():
    """Runs a sim with several materials and checks that the produced value is
    not zero at the centre
    """
    # build
    parameters = {}
    mat1 = {
        "E_D": 0,
        "D_0": 1,
        "id": [1, 2]
    }

    mat2 = {
        "E_D": 0,
        "D_0": 0.25,
        "id": 3
    }

    parameters["materials"] = [mat1, mat2]

    N = 16
    mesh = fenics.UnitSquareMesh(N, N)
    vm = fenics.MeshFunction("size_t", mesh, 2, 0)
    sm = fenics.MeshFunction("size_t", mesh, 1, 0)

    tol = 1E-14
    subdomain_1 = fenics.CompiledSubDomain('x[1] <= 0.5 + tol', tol=tol)
    subdomain_2 = fenics.CompiledSubDomain('x[1] >= 0.5 - tol && x[0] >= 0.5 - tol', tol=tol)
    subdomain_3 = fenics.CompiledSubDomain('x[1] >= 0.5 - tol && x[0] <= 0.5 + tol', tol=tol)
    subdomain_1.mark(vm, 1)
    subdomain_2.mark(vm, 2)
    subdomain_3.mark(vm, 3)

    surfaces = fenics.CompiledSubDomain('on_boundary')
    surfaces.mark(sm, 1)

    parameters["mesh_parameters"] = {
        "mesh": mesh,
        "volume_markers": vm,
        "surface_markers": sm,
    }

    parameters["traps"] = [
    ]

    parameters["temperature"] = {
        "type": "expression",
        "value": 30,
    }
    parameters["source_term"] = {
        "type": "expression",
        "value": 1,
    }
    parameters["boundary_conditions"] = [
        {
            "type": "dc",
            "value": 0,
            "surfaces": 1
        }

    ]
    parameters["exports"] = {
    }
    solving_parameters = {
        "type": "solve_stationary",
        "newton_solver": {
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-10,
            "maximum_iterations": 5,
        }
    }

    parameters["solving_parameters"] = solving_parameters

    # run
    my_sim = FESTIM.Simulation(parameters, log_level=20)
    my_sim.initialise()
    my_sim.run()

    # test

    assert my_sim.u(0.5, 0.5) != 0


def test_steady_state_traps_not_everywhere():
    """Creates a simulation problem with a trap not set in all subdomains runs
    the sim and check that the value is not NaN
    """
    parameters = {}
    mat1 = {
        "borders": [0, 0.25],
        "E_D": 0,
        "D_0": 1,
        "id": 1
    }
    mat2 = {
        "borders": [0.25, 0.5],
        "E_D": 0,
        "D_0": 1,
        "id": 2
    }
    mat3 = {
        "borders": [0.5, 1],
        "E_D": 0,
        "D_0": 1,
        "id": 3
    }
    parameters["materials"] = [mat1, mat2, mat3]

    parameters["mesh_parameters"] = {
        "initial_number_of_cells": 100,
        "size": 1
    }

    trap = {
        "k_0": 1,
        "E_k": 0,
        "p_0": 1,
        "E_p": 0,
        "density": 1,
        "materials": [1, 3]
    }
    parameters["traps"] = [trap]
    parameters["temperature"] = {
        "type": "expression",
        "value": 1,
    }
    parameters["boundary_conditions"] = [
        {
            "type": "dc",
            "value": 1,
            "surfaces": 1
        }
    ]
    parameters["exports"] = {}
    solving_parameters = {
        "type": "solve_stationary",
        "traps_element_type": "DG",
        "newton_solver": {
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-10,
            "maximum_iterations": 5,
        }
    }

    parameters["solving_parameters"] = solving_parameters

    my_sim = FESTIM.Simulation(parameters)
    my_sim.initialise()
    my_sim.run()
    assert not np.isnan(my_sim.u.split()[1](0.5))


def test_no_jacobian_update():
    """Runs a transient sim and with the flag "update_jacobian" set to False.
    """

    parameters = {
        "materials": [
            {
                "E_D": 0,
                "D_0": 1,
                "id": 1
                }
                ],
        "traps": [
            ],
        "initial_conditions": [
        ],
        "mesh_parameters": {
                "initial_number_of_cells": 10,
                "size": 1,
            },
        "boundary_conditions": [
            ],
        "temperature": {
                'type': "expression",
                'value': 300
            },
        "source_term": {
            'value': 1
            },
        "solving_parameters": {
            "final_time": 10,
            "initial_stepsize": 1,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1,
                },
            "newton_solver": {
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 50,
            },
            "update_jacobian": False
            },
        "exports": {
            },
    }

    FESTIM.run(parameters)


def test_nb_iterations_bewteen_derived_quantities_compute(tmpdir):
    """Checks that "nb_iterations_between_compute" has an influence on the
    number of entries in derived quantities
    """
    d = tmpdir.mkdir("temp")
    parameters = {
        "materials": [
            {
                "E_D": 1,
                "D_0": 1,
                "id": 1
            }
            ],
        "traps": [
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 200,
                "size": 1,
                "refinements": [
                ],
            },
        "boundary_conditions": [
            ],
        "temperature": {
            "type": "expression",
            "value": 300
        },
        "solving_parameters": {
            "type": "solve_transient",
            "final_time": 30,
            "initial_stepsize": 4,
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 50,
            }
            },
        "exports": {
            "derived_quantities": {
                "total_volume": [{
                    "field": 'retention',
                    "volumes":  [1],
                }],
                "folder": str(Path(d)),
                "nb_iterations_between_compute": 2
            },
            },
    }
    output = FESTIM.run(parameters)
    short_derived_quantities = output["derived_quantities"]

    parameters["exports"]["derived_quantities"]["nb_iterations_between_compute"] = 1
    output = FESTIM.run(parameters)
    long_derived_quantities = output["derived_quantities"]

    assert len(long_derived_quantities) > len(short_derived_quantities)


def test_nb_iterations_bewteen_derived_quantities_export(tmpdir):
    """Checks that a simulation with "nb_iterations_between_exports" key for
    derived quantities doesn't raise an error
    """
    d = tmpdir.mkdir("temp")
    parameters = {
        "materials": [
            {
                "E_D": 1,
                "D_0": 1,
                "id": 1
            }
            ],
        "traps": [
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 200,
                "size": 1,
                "refinements": [
                ],
            },
        "boundary_conditions": [
            ],
        "temperature": {
            "type": "expression",
            "value": 300
        },
        "solving_parameters": {
            "type": "solve_transient",
            "final_time": 30,
            "initial_stepsize": 4,
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 50,
            }
            },
        "exports": {
            "derived_quantities": {
                "total_volume": [{
                    "field": 'retention',
                    "volumes":  [1],
                }],
                "folder": str(Path(d)),
                "nb_iterations_between_exports": 2
            },
            },
    }
    output = FESTIM.run(parameters)


def test_error_steady_state_diverges():
    """Checks that when a sim doesn't converge in steady state, an error is
    raised
    """
    parameters = {
        "materials": [
            {
                "E_D": 1,
                "D_0": 1,
                "id": 1
                }
                ],
        "traps": [
            ],
        "initial_conditions": [
        ],
        "mesh_parameters": {
                "initial_number_of_cells": 10,
                "size": 1,
            },
        "boundary_conditions": [
            ],
        "temperature": {
                'type': "expression",
                'value': -1
            },
        "solving_parameters": {
            "type": "solve_stationary",
            "newton_solver": {
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 2,
            },
        },
        "exports": {
        },
    }
    with pytest.raises(ValueError) as err:
        FESTIM.run(parameters)
    assert "The solver diverged" in str(err.value)
