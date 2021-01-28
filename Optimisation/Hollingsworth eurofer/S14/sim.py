from context3 import context2
from context2 import context
from context import FESTIM
from FESTIM import *
from FESTIM.generic_simulation import run

flux = 3e17
implantation_time = 1e23/flux
resting_time = 24*3600
ramp = 10/60
tds_time = (1000 - 300)/ramp


def simu(p):
    '''
    Runs the simulation with parameters p
    '''
    file_name = "desorption-"
    for e in p:
        file_name += ';' + str(e)

    density = 7.798e6/55.845*6.022e23
    size = 2e-3
    r = 0
    center = 6e-9
    width = 2.5e-9
    distribution = 1/(width*(2*3.14)**0.5) * sp.exp(-0.5*((x-center)/width)**2)
    parameters = {
        "materials": [
            {
                "alpha": 5.6605e-10,  # lattice constant ()
                "beta": density,  # number of solute sites per atom (6 for W)
                "borders": [0, size],
                "E_diff": 30.4e3*FESTIM.k_B/FESTIM.R,
                "D_0": 1.33e-6,
                "id": 1
                }
                ],
        "traps": [

            {
               "energy": p[0],
               "density": p[1] * 1e-2 * density,
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
                'value': (50+273.15) * (t < implantation_time) + \
                        300 * (t >= implantation_time) * \
                        (t < implantation_time + resting_time) + \
                        (t > implantation_time + resting_time) * \
                        (300 + ramp * (t - (implantation_time + resting_time)))
            },
        "source_term": {
            'value': flux * (1 - r) * distribution * (t <= implantation_time)
            },
        "solving_parameters": {
            "final_time": implantation_time + resting_time + tds_time,
            "initial_stepsize": 0.5,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1.1,
                "t_stop": implantation_time + resting_time - 0,
                "stepsize_stop_max": 30,
                "dt_min": 1e-5
                },
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 50,
            }
            },
        "exports": {
            "txt": {
                'times': [implantation_time + resting_time - 0],
                'functions': ["retention"],
                'labels': ["retention"],
                "folder": "Results"
            },

            "derived_quantities": {
                # "file": '_'.join(map(str, p)) + ".csv",
                "file": "last_unpondered.csv",
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
            #     "functions": ["solute", "1", "retention"],
            #     "labels": ["solute", "1", "retention"],
            #     "folder": "Results",
            # }
            }
            }

    output = run(parameters, log_level=30)
    return output["derived_quantities"]


if __name__ == "__main__":
    p = [1.1, 2.6]
    simu(p)
