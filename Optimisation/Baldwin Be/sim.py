from context2 import context
from context import FESTIM
from FESTIM import *
from FESTIM.generic_simulation import run


implantation_time = 7200
resting_time = 100
ramp = 0.3
tds_time = (800-300)/ramp

T_D2 = 300

T = (400) * (t < implantation_time) + \
        (400 - (1-sp.exp(-(t - implantation_time)/2700))*100) * (t >= implantation_time) * \
        (t <= implantation_time + resting_time) + \
        (t > implantation_time + resting_time) * \
        (300 + ramp * (t - (implantation_time + resting_time)))
P_0 = 1*0.8
P = P_0 * (FESTIM.t < implantation_time)

# M_D2 = 1
# value = P/((2*3.14*M_D2*FESTIM.k_B*T_D2)**0.5)


def simu(p):
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
                "density": 0.101*density,
                "energy": 0.805,
                "materials": 1
            },
            {
                "density": 0.032*density,
                "energy": 1.070,
                "materials": 1
            }
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 500,
                "size": size,
                "refinements": [
                    {
                        "cells": 100,
                        "x": 30e-9
                    },
                ],
            },
        "boundary_conditions": [
                # {
                #     "surfaces": [1],
                #     "value": value,
                #     "type": "flux"
                # },
                # {
                #     "surfaces": [1],
                #     "Kr_0": 3.4e-29,
                #     "E_Kr": 0.280,
                #     "type": "recomb"
                # },
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
            "final_time": implantation_time + resting_time + tds_time,
            "initial_stepsize": 1,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1.1,
                "t_stop": implantation_time + resting_time - 10,
                "stepsize_stop_max": 7,
                "dt_min": 1e-5
                },
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 50,
            }
            },
        "exports": {
            "txt": {
                'times': [implantation_time, implantation_time + resting_time],
                'functions': ["solute"],
                'labels': ["solute"],
                "folder": "Results"
            },

            "derived_quantities": {
                # "file": '_'.join(map(str, p)) + ".csv",
                "file": "last.csv",
                "folder": "Results/derived_quantities",
                "average_volume": [
                    {
                        "field": "T",
                        "volumes": [1]
                    }
                ],
                # "surface_flux": [
                #     {
                #         "field": "solute",
                #         "surfaces": [1, 2]
                #     }
                # ],

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
            "xdmf": {
                "functions": ["solute", "T", "1", "2", "retention"],
                "labels": ["solute", "T", "1", "2", "retention"],
                "folder": "Results",
            }
            }
            }

    output = run(parameters, log_level=30)
    return output["derived_quantities"]


if __name__ == "__main__":
    p = [1, 0.89]
    simu(p)
