from context2 import context
from context import FESTIM
from FESTIM import *
from FESTIM.generic_simulation import run

flux = 1.7e20
implantation_time = 3e23/flux
resting_time = 100
tds_time = 600


def simu(p):
    '''
    Runs the simulation with parameters p
    '''
    file_name = "desorption-"
    for e in p:
        file_name += ';' + str(e)

    density = 2.7e6/26.982*6.022e23
    size = 2e-3
    r = 0
    center = 10e-9
    width = 2.5e-9
    distribution = 1/(width*(2*3.14)**0.5) * sp.exp(-0.5*((x-center)/width)**2)
    parameters = {
        "materials": [
            {
                "alpha": 5.6605e-10,  # lattice constant ()
                "beta": density,  # number of solute sites per atom (6 for W)
                "borders": [0, size],
                "E_diff": 0.52,
                "D_0": 2.6e-5,
                "id": 1
                }
                ],
        "traps": [

            {
               "energy": p[0],
               "density": p[1] * 1e-2* density,
               "materials": 1,
            },
            # {
            #   "energy": p[2],
            #   "density": p[3] * density,
            #   "materials": 1,
            # },
            # {
            #   "energy": p[4],
            #   "density": p[5] * density,
            #   "materials": 1,
            # },
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 500,
                "size": size,
                "refinements": [
                    {
                        "cells": 300,
                        "x": 3e-6
                    },
                    {
                        "cells": 50,
                        "x": 30e-9
                    }
                ],
            },
        "boundary_conditions": [
                {
                    "surface": [1, 2],
                    "value": 0,
                    "component": 0,
                    "type": "dc"
                }
            ],
        "temperature": {
                'type': "expression",
                'value': 618 * (t < implantation_time) + \
                        300 * (t >= implantation_time) * \
                        (t < implantation_time + resting_time) + \
                        (t > implantation_time + resting_time) * \
                        (300 + 1 * (t - (implantation_time + resting_time)))
            },
        "source_term": {
            'value': flux * (1 - r) * distribution * (t <= implantation_time)
            },
        "solving_parameters": {
            "final_time": implantation_time + resting_time + tds_time,
            "initial_stepsize": 0.5,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1.1,
                "t_stop": implantation_time + resting_time - 100,
                "stepsize_stop_max": 3,
                "dt_min": 1e-5
                },
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 50,
            }
            },
        "exports": {
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
                        "field": "retention",
                        "volumes": [1]
                    },
                    {
                        "field": "solute",
                        "volumes": [1]
                    },
                    {
                        "field": "1",
                        "volumes": [1]
                    },
                    # {
                    #     "field": "2",
                    #     "volumes": [1]
                    # },
                    # {
                    #     "field": "3",
                    #     "volumes": [1]
                    # },
                    ],
            },
            # "xdmf": {
            #     "functions": ["solute", "1", "2", "3", "retention"],
            #     "labels": ["solute", "1", "2", "3", "retention"],
            #     "folder": "Results",
            # }
            }
            }

    output = run(parameters, log_level=30)
    return output["derived_quantities"]


if __name__ == "__main__":
    p = [1.1, 1]
    simu(p)
