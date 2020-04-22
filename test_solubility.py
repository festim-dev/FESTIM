import FESTIM
from FESTIM.generic_simulation import run

# atom_density  =  density(g/m3)*Na(/mol)/M(g/mol)
atom_density_W = 6.28e28   #  6.3222e28  # atomic density m^-3
atom_density_Cu = 8.43e28   # 8.4912e28  # atomic density m^-3
atom_density_CuCrZr = 8.43e28   #  2.6096e28  # atomic density m^-3

# IDs for edges and surfaces (must be the same as in xdmf files)
id_W = 8
id_Cu = 7
id_CuCrZr = 6

id_top_surf = 1  # 9
id_coolant_surf = 2  # 10
# id_left_surf = 11

folder = "solution_test"


# OK TMAP
def rhoCp_W(T):
    return 5.15356e-6*T**3-8.30703e-2*T**2+5.98312e2*T+2.48160e6


# OK TMAP
def thermal_cond_W(T):
    return -7.84154e-9*T**3+5.03006e-5*T**2-1.07335e-1*T+1.75214e2


# OK TMAP
def rhoCp_Cu(T):
    return 1.68402e-4*T**3+6.14079e-2*T**2+4.67353e2*T+3.45899e6


# OK TMAP
def thermal_cond_Cu(T):
    return -7.84154e-9*T**3+5.03006e-5*T**2-1.07335e-1*T+1.75214e2


# OK TMAP
def rhoCp_CuCrZr(T):
    return -1.79134e-4*T**3+1.51383e-1*T**2+6.22091e2*T+3.46007e6


# OK TMAP
def thermal_cond_CuCrZr(T):
    return 5.25780e-7*T**3-6.45110e-4*T**2+2.57678e-01*T+3.12969e2


size = 8.5e-3  # OK TMAP
parameters = {
    "mesh_parameters": {
        "size": size,
        "initial_number_of_cells": 15000,
        "refinements": [
            {
                "cells": 500,
                "x": 3e-6,
            },
            {
                "cells": 100,
                "x": 30e-9,
            },
            ]
        },
    "materials": [
        {
            # Tungsten
            "D_0": 2.9e-7*0.8165,  # OK TMAP eq(6)
            "E_D": 0.39,  # OK TMAP eq(6)
            # "S_0": 1.87e24,  # case without solubility
            # "E_S": 1.04,  # case without solubility
            "thermal_cond": thermal_cond_W,
            "heat_capacity": 1,
            "rho": rhoCp_W,
            "borders": [0, size],  # OK TMAP
            "id": id_W,
        },
        ],
    "traps": [
        ],
    "source_term": [
       {
            'value': 5e23 * 1/2.5e-9 * (FESTIM.x < 2.5e-9),  #* (FESTIM.t <= 400)
            'volumes': [id_W]
       },
       ],    
    "boundary_conditions": [
        {
           "type": "recomb",
           "surfaces": 1,
            "Kr_0": 3.2e-15,
            "E_Kr": 1.16,
            "order": 2,
        },
        {
            "type": "dc",
            "surfaces": 2,
            "value": 0
        },
        ],
    "temperature": {
        "type": "expression",
        "value": 1000
        },
    "solving_parameters": {
        # "type": "solve_stationary",
        "final_time": 2, # OK TMAP ligne 183
        "initial_stepsize": 0.0001,
        "adaptive_stepsize": {
            "stepsize_change_ratio": 1.15,
            "t_stop": 1e8,
            "stepsize_stop_max": 1e7,
            "dt_min": 1e-8,
            },
        #"times": [],
        "newton_solver": {
            "absolute_tolerance": 1e11,
            "relative_tolerance": 1e-10,
            "maximum_iterations": 20,
        }
        },
    "exports": {
         "xdmf": {
             "functions": ['0'],
             "labels": ['solute',],
             "folder": folder,
             "all_timesteps": True,
        },
    }

}
run(parameters, log_level=20)
