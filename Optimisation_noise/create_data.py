from context import FESTIM
from FESTIM import *
from FESTIM.generic_simulation import run
import sympy as sp
import numpy as np
import csv


folder = 'experimental data'
implantation_time = 400
resting_time = 50
tds_time = 100
density = 6.3e28
size = 2e-5
center = 4.5e-9
width = 2.5e-9
distribution = 1/(width*(2*3.14)**0.5) * sp.exp(-0.5*((x-center)/width)**2)
parameters = {
    "materials": [
        {
            "alpha": 1.1e-10,  # lattice constant ()
            "beta": 6 * density,  # number of solute sites per atom (6 for W)
            "borders": [0, size],
            "E_diff": 0.39,
            "D_0": 4.1e-7,
            "id": 1
            }
            ],
    "traps": [
        {
          "energy": 0.87,
          "density": 1.3e-3 * density,
          "materials": 1,
        },
        {
          "energy": 1.1,
          "density": 0.5e-3 * density,
          "materials": 1,
        },
        # {
        #   "energy": 1.5,
        #   "materials": 1,
        #   "type": 'extrinsic',
        #   "form_parameters":{
        #       "phi_0": 2.5e19*(t<=implantation_time),
        #       "n_amax": 1e-1*density,
        #       "f_a": distribution,
        #       "eta_a": 6e-4,
        #       "n_bmax": 1e-2*density,
        #       "f_b": (x<1e-6)*(x>0)*(1/1e-6),
        #       "eta_b": 2e-4
        #   }
        # },
        ],
    "mesh_parameters": {
            "initial_number_of_cells": 200,
            "size": size,
            "refinements": [
                {
                    "cells": 300,
                    "x": 3e-6
                },
                {
                    "cells": 100,
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
            'value': 300 + (t > implantation_time + resting_time) * 4 * (t - (implantation_time + resting_time))
        },
    "source_term": {
        'value': 2.5e19 * distribution * (t <= implantation_time)
        },
    "solving_parameters": {
        "final_time": implantation_time + resting_time + tds_time,
        "initial_stepsize": 0.5,
        "adaptive_stepsize": {
            "stepsize_change_ratio": 1.1,
            "t_stop": implantation_time + resting_time - 20,
            "stepsize_stop_max": 0.1,
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
            "functions": [],
            "times": [],
            "labels": [],
            "folder": folder
            },
        #"xdmf": {
        #   "functions": ['solute', '1', '2'],
        #   "labels":  ['solute', 'trap1', 'trap2'],
        #   "folder": folder
        #},
        "TDS": {
            "file": "data",
            "TDS_time": implantation_time + resting_time,
            "folder": folder
            },
        }
        }
output = run(parameters)
