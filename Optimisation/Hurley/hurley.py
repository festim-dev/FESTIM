from context import FESTIM
from FESTIM.generic_simulation import run
import sympy as sp
import numpy as np


def hurley_tds(charging_type, ramp, folder):

    density = 8.4897e28
    # TS_density = 2.47e-5 * density
    TS_density = 2e24

    size = 0.2e-2/(1+1*(charging_type == "A"))
    implantation_time = 90*60*(charging_type == "A") + 3*60*(charging_type == "B")
    resting_time = 30*60
    tds_time = 6000

    parameters = {
        "materials": [
            {
                "alpha": 1,
                # "beta":  7.2e-4*1e-4*density/2.64e7,
                "beta":  7.2e-4*1e-4/4.4e-23,
                "borders": [0, size],
                # "E_diff": 0.39,
                # "D_0": 4.1e-7,
                "E_diff": 5.69e3*FESTIM.k_B/FESTIM.R,
                "D_0": 7.2e-4*1e-4,
                "id": 1
                }
                ],
        "traps": [
            {
                "energy": 0.511601461,
                "density": 2.02271302e24,
                "materials": 1,
            },
            {
                "energy": 0.56699002,
                "density": 1.03998491e24,
                "materials": 1,
            },
            # {
            #     "energy": 0.5151,
            #     "density": 2.1253e24,
            #     "materials": 1,
            # },
            ],
        "mesh_parameters": {
                "initial_number_of_cells": int(1200/(1+1*(charging_type == "A"))),
                "size": size,
                "refinements": [
                    # {
                    #     "cells": 3000,
                    #     "x": 0.5e-2
                    # },
                    {
                        "cells": 600,
                        "x": 3e-6
                    },
                    # {
                    #     "cells": 12000,
                    #     "x": 30e-8
                    # },
                    {
                        "cells": 100,
                        "x": 50e-9
                    },
                ],
            },
        "boundary_conditions": [
                {
                    "surface": [1],
                    # "value": 1.1e-6 * density * (FESTIM.t <= implantation_time),
                    "value": 0.045 * TS_density * (FESTIM.t <= implantation_time),
                    "component": 0,
                    "type": "dc"
                },
            ],
        "temperature": {
                'type': "expression",
                'value': 298 + (FESTIM.t > implantation_time + resting_time) * ramp/60 * (FESTIM.t - (implantation_time + resting_time))
            },
        "solving_parameters": {
            "final_time": implantation_time + resting_time + tds_time,
            "initial_stepsize": 0.0001,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1.1,
                "t_stop": implantation_time + resting_time - 20,
                "stepsize_stop_max": 10,
                "dt_min": 1e-7,
                },
            "newton_solver": {
                "absolute_tolerance": 1e12,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 10,
            }
            },
        "exports": {
            "xdmf": {
                "functions": ["solute", "1", "2", "retention"],
                "labels": ["solute", "trapped_1", "trapped_2", "retention"],
                "folder": folder
            },
            "txt": {
                "times": [implantation_time, implantation_time+resting_time],
                "functions": [1],
                "labels": ["trapped"]
            },
            "derived_quantities": {
                "file": "derived_quantities.csv",
                "folder": folder,
                "average_volume": [
                    {
                        "field": "T",
                        "volumes": [1]
                    }
                ],
                "surface_flux": [
                    {
                        "field": "solute",
                        "surfaces": [1, 2]
                    },
                ],
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
                    {
                        "field": "2",
                        "volumes": [1]
                    },
                    # {
                    #     "field": "3",
                    #     "volumes": [1]
                    # },
                    ],
            }
            }
        }
    if charging_type == "B":
        parameters["boundary_conditions"].append(
            {"surface": [2], "value": 0,
             "component": 0, "type": "dc"})
    output = run(parameters, log_level=30)
    return output


ramp = 2
charging_type = "B"
folder = "test/2_traps/type_" + charging_type + "/beta=" + str(ramp) + "Kmin-1"

hurley_tds(charging_type, ramp, folder)
