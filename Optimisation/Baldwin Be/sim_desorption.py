from context2 import context
from context import FESTIM
from FESTIM import *
from FESTIM.generic_simulation import run


ramp = 0.3
resting_time = 306
tds_time = (900-300)/ramp

T_D2 = 300

T = 300 + ramp * (t - resting_time) * (FESTIM.t > resting_time)

P = 1.33e-6


def simu_recomb(p):
    '''
    Runs the simulation with parameters p
    '''
    file_name = "desorption-"
    for e in p:
        file_name += ';' + str(e)

    density = 1.85e6/9.0122*6.022e23
    size = 1e-6
    parameters = {
        "materials": [
            {
                "alpha": 1,
                "beta": 8e-9/(4e12)*density,
                "E_diff": 0.364,
                "D_0": 8e-9,
                "borders": [0, size],
                "id": 1
                },
                ],
        "traps": [
            {
                "density": p[0]*density,
                "energy": p[1],
                "materials": 1
            },
            {
                "density": p[3]*density,
                "energy": p[4],
                "materials": 1
            }
            ],
        "initial_conditions": [
            # {
            #     "value": 0,
            #     "component": 0
            # },
            {
                "value": p[0]*density*p[2],
                "component": 1
            },
            {
                "value": p[3]*density*p[5],
                "component": 2
            }
        ],
        "mesh_parameters": {
                "initial_number_of_cells": 600,
                "size": size,
                "refinements": [
                    # {
                    #     "cells": 100,
                    #     "x": 30e-8
                    # },
                ],
            },
        "boundary_conditions": [

                {
                    "surface": [1],
                    "Kr_0": 3.4e-29,
                    "E_Kr": 0.280,
                    "order": 2,
                    "type": "recomb"
                },
                {
                    "surface": [1],
                    "type": "flux",
                    "value": P/((2*3.14*6.68e-27*FESTIM.k_B*T_D2)**0.5)
                }
            ],
        "temperature": {
                'type': "expression",
                'value': T
            },
        "solving_parameters": {
            "final_time": resting_time + tds_time,
            "initial_stepsize": 0.1,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1.2,
                "t_stop": resting_time,
                "stepsize_stop_max": 15,
                "dt_min": 1e-10
                },
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 50,
            }
            },
        "exports": {

            "derived_quantities": {
                # "file": '_'.join(map(str, p)) + ".csv",
                "file": "last_recomb.csv",
                "folder": "Results/derived_quantities",

                "total_volume": [
                    {
                        "field": "solute",
                        "volumes": [1]
                    },
                    {
                        "field": "1",
                        "volumes": [1]
                    },
                    {
                        "field": "2",
                        "volumes": [1]
                    },
                    {
                        "field": "retention",
                        "volumes": [1]
                    },
                    ],
            },
            # "xdmf": {
            #     "functions": ["solute", "T", "1", "2", "retention"],
            #     "labels": ["solute", "T", "1", "2", "retention"],
            #     "folder": "Results",
            # }
            }
            }

    output = run(parameters, log_level=30)
    return output["derived_quantities"]


def simu_sievert(p):
    '''
    Runs the simulation with parameters p
    '''
    file_name = "desorption-"
    for e in p:
        file_name += ';' + str(e)

    density = 1.85e6/9.0122*6.022e23
    size = 1e-6
    parameters = {
        "materials": [
            {
                "alpha": 1,
                "beta": 8e-9/(4e12)*density,
                "E_diff": 0.364,
                "D_0": 8e-9,
                "borders": [0, size],
                "id": 1
                },
                ],
        "traps": [
            {
                "density": p[0]*density,
                "energy": p[1],
                "materials": 1
            },
            {
                "density": p[3]*density,
                "energy": p[4],
                "materials": 1
            }
            ],
        "initial_conditions": [
            # {
            #     "value": 0,
            #     "component": 0
            # },
            {
                "value": p[0]*density*p[2],
                "component": 1
            },
            {
                "value": p[3]*density*p[5],
                "component": 2
            }
        ],
        "mesh_parameters": {
                "initial_number_of_cells": 600,
                "size": size,
                "refinements": [
                    # {
                    #     "cells": 100,
                    #     "x": 30e-8
                    # },
                ],
            },
        "boundary_conditions": [
                {
                    "surface": [1],
                    "value": 2.3e22*sp.exp(-0.174/FESTIM.k_B/T)*P**0.5,
                    "type": "dc"
                },
            ],
        "temperature": {
                'type': "expression",
                'value': T
            },
        "solving_parameters": {
            "final_time": resting_time + tds_time,
            "initial_stepsize": 10,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1.2,
                "t_stop": resting_time,
                "stepsize_stop_max": 15,
                "dt_min": 1e-10
                },
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 50,
            }
            },
        "exports": {

            "derived_quantities": {
                # "file": '_'.join(map(str, p)) + ".csv",
                "file": "last_sievert.csv",
                "folder": "Results/derived_quantities",

                "total_volume": [
                    {
                        "field": "solute",
                        "volumes": [1]
                    },
                    {
                        "field": "1",
                        "volumes": [1]
                    },
                    {
                        "field": "2",
                        "volumes": [1]
                    },
                    {
                        "field": "retention",
                        "volumes": [1]
                    },
                    ],
            },
            # "xdmf": {
            #     "functions": ["solute", "T", "1", "2", "retention"],
            #     "labels": ["solute", "T", "1", "2", "retention"],
            #     "folder": "Results",
            # }
            }
            }

    output = run(parameters, log_level=30)
    return output["derived_quantities"]


if __name__ == "__main__":
    # p = [0.101, 0.805, 0.7, 0.032, 1.07, 0.3]
    # p = [0.098, 0.743, 0.7, 0.032, 0.94, 0.3]
    p = [0.098, 0.743, 0.7, 0.032, 0.94, 0.3]
    simu(p)
