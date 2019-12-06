from context import FESTIM
import sympy as sp


# Storage folder
folder = 'results/05_ITER_case_theta_sol2_99950/'

# Dict parameters
parameters = {
    "mesh_parameters": {
        "size": 1,
        "initial_number_of_cells": 20,
        "refinements": []
        },
    "materials": [
        {
            "D_0": 2.9e-7,
            "E_diff": 0.39,
            "S_0": 1,  # at/m3.Pa0.5 (from Grislia 2015)
            "E_S": 0.34,  # eV
            #"S_0": 1.0,  # case without solubility
            #"E_S": 0.0,  # case without solubility
            "alpha": 1,  # 1.0000 coef H/D/T
            "beta": 1,
            "borders": [0, 0.5],
            "id": 1,
        },
        {
            "D_0": 6.6e-7,
            "E_diff": 0.387,
            "S_0": 3.12e28,  # at/m3.Pa0.5 (from ITER)
            "E_S": 0.572,  # eV
            #"S_0": 1.0,  # case without solubility
            #"E_S": 0.0,  # case without solubility
            "alpha": 1,
            "beta": 1,
            "borders": [0.5, 1],
            "id": 2,
        },
        ],
    "traps": [
        ],
    "boundary_conditions": [
        {
            "type": "dc",
            "surface": 2,
            "value": 4
        },
        {
            "type": "dc",
            "surface": 1,
            "value": 4
        },
        ],
    "temperature": {
        "type": "expression",
        "value": 300
        },
    "solving_parameters": {
        "final_time": 1,
        "times": [],
        "initial_stepsize": 0.1,
        "adaptive_stepsize": {
            "stepsize_change_ratio": 1,
            "t_stop": 1,
            "stepsize_stop_max": 0.1,
            "dt_min": 1e-8,
            },
        "newton_solver": {
            "absolute_tolerance": 1e10,
            "relative_tolerance": 1e-9,
            "maximum_iterations": 10,
        }
        },
    "exports": {
        "xdmf": {
            "functions": ['0'],
            "labels": ['theta'],
            "folder": folder,
            "all_timesteps": False,
        },
    }
}
