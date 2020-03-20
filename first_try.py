from FESTIM import *
from FESTIM.generic_simulation import *

# 1st step: create an empty dict
parameters = {}

# Mesh creation
mesh_parameters = {
            "size": 2e-5,                       # length of the slab
            "initial_number_of_cells": 200,      # first broad mesh
            "refinements": [
                    {
                        "cells": 300,           # with N cells
                        "x" : 3e-6,             # first refinement within the first x m
                    },
                    {
                        "cells": 120,           # with the N cells
                        "x": 30e-9,              # second refinement within the first x m
                    }
                ],
            }
parameters["mesh_parameters"] = mesh_parameters # ajout dans le dictionnaire

# MATERIAL PROPERTIES
material = [{
            "borders":[0, 2e-5], 
            "E_D": 0.20,            
            "D_0": 1.9e-7,
            "id": 1
            }]
parameters["materials"] = material

# SOURCE VOLUMETRIC
import sympy as sp
import math

center = 4.5e-9
width = 2.5e-9
distribution = 1.0/(width*(2*math.pi)**0.5) * sp.exp(-0.5*((x-center)/width)**2)

source_term = {
        "value" : 2.5e19 *distribution* (t<= 400)   #flux*distribution*t_implantation
        }
parameters["source_term"] = source_term

# TRAPPING PARAMETER
traps = [
        {
            "k_0": 1.9e-7/(1.1e-10**2*6.0*6.3e28),
            "E_k": 0.20,
            "p_0": 1e13,
            "E_p": 0.87,
            "density": 1.3e-3*6.3e28,
            "materials": [1]
        },
        {
            "k_0": 1.9e-7/(1.1e-10**2*6.0*6.3e28),
            "E_k": 0.20,
            "p_0": 1e13,
            "E_p": 1.00,
            "density": 4.0e-4*6.3e28,
            "materials": [1]
        },
        {
            "k_0": 1.9e-7/(1.1e-10**2*6.0*6.3e28),
            "E_k": 0.20,
            "p_0": 1e13,
            "E_p": 1.50,
            "materials": [1],
            "type": "extrinsic",
            "form_parameters":{
                "phi_0":2.5e19* (t <= 400),
                "n_amax": 1e-1*6.3e28,
                "f_a": distribution,
                "eta_a": 6e-4,
                "n_bmax": 1e-6,
                "f_b": (x<1e-6)*(x>0)*(1/1e-6),
                "eta_b":0,
            }
        }
        ]
parameters["traps"] = traps

boundary_conditions = [
            {
                "surfaces": [1],
                "value": 0,
                "component": 0,
                "type": "dc"
            },
            {
                "surfaces": [2],
                "value": 0,
                "type": "dc"
            }
    ]

parameters["boundary_conditions"] = boundary_conditions

temperature = {
        "type" : "expression",
        "value" : 300 + (t>450) * (8*(t-450))
        }
parameters["temperature"] = temperature

solving_parameters = {
        "final_time":500,
        "initial_stepsize":0.5,
        "adaptive_stepsize":{
            "stepsize_change_ratio":1.1,
            "t_stop":430,
            "stepsize_stop_max":0.5,
            "dt_min":1e-5
            },
        "newton_solver": {
            "absolute_tolerance": 1e10,
            "relative_tolerance": 1e-9,
            "maximum_iterations": 50,
            }
        }

parameters["solving_parameters"] = solving_parameters

# EXPORT OUTPUT
folder = "Solution_ogorodnikova"

exports = {
        "txt" : {
            "functions": ["retention"],
            "times":[],
            "labels":["retention"],
            "folder":folder
            },
        "xdmf": {
            "functions":["solute","1","2","3","retention"],
            "labels": ["solute","trap_1","trap_2","trap_3","retention"],
            "folder":folder
            }
        }
parameters["exports"]=exports

for key, value in parameters.items():
    print(str(key)+":"+str(value))
    print("\n")

output = run(parameters)

print(output.keys())

import numpy as np
np.set_printoptions(threshold=20)
#print(np.array(output["temperature"]))
print(np.array(output["derived_quantities"]))

