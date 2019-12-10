from context import FESTIM
import sympy as sp


# Storage folder
folder = 'results/'

# Dict parameters
T = 300 * (FESTIM.t < 0.1) + 300
parameters = {
    "mesh_parameters": {
        "size": 1,
        "initial_number_of_cells": 200,
        "refinements": [],
        },
    "materials": [
        {
            "D_0": 1,
            "E_diff": 0,
            "S_0": 1,  # at/m3.Pa0.5 (from Grislia 2015)
            "E_S": 0.1,  # eV
            "alpha": 1,  # 1.0000 coef H/D/T
            "beta": 1,
            "borders": [0, 0.5],
            "id": 1,
        },
        {
            "D_0": 1,
            "E_diff": 0,
            "S_0": 1,  # at/m3.Pa0.5 (from ITER)
            "E_S": 0.1,  # eV
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
            "surface": 1,
            "value": 4/(1*sp.exp(-0.1/FESTIM.k_B/T))
        },
        {
            "type": "dc",
            "surface": 2,
            "value": 4/(1*sp.exp(-0.1/FESTIM.k_B/T))
        },
        ],
    "temperature": {
        "type": "expression",
        "value": T
        },
    "solving_parameters": {
        "type": "solve_transient",
        "final_time": 0.2,
        "times": [15],
        "initial_stepsize": 0.001,
        "adaptive_stepsize": {
            "stepsize_change_ratio": 1,
            "t_stop": 111,
            "stepsize_stop_max": 0.1,
            "dt_min": 1e-8,
            },
        "newton_solver": {
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-9,
            "maximum_iterations": 10,
        }
        },
    "exports": {
        "xdmf": {
            "functions": ['0', 'retention'],
            "labels": ['theta', 'solute'],
            "folder": folder,
            "all_timesteps": True,
        },
    }
}
