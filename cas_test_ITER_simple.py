import FESTIM
from fenics import *
import sympy as sp


def bc_top_H(t_implantation, t_rest, t_baking):
    t = FESTIM.t
    implantation = (t < t_implantation) * 1e23*2.5e-9/(2.9e-7*sp.exp(-0.39/FESTIM.k_B/1200))
    expression = implantation

    return expression


def bc_top_HT(t_implantation, t_rest, t_baking):
    t = FESTIM.t
    implantation = (t < t_implantation) * 1200
    rest = (t > t_implantation)*(t < t_implantation + t_rest) * 343
    baking = (t > t_implantation + t_rest)*(t < t_implantation + t_rest + t_baking)*350
    expression = implantation + rest + baking
    return expression


def bc_coolant_HT(t_implantation, t_rest, t_baking):
    t = FESTIM.t
    implantation = (t < t_implantation) * 373
    rest = (t > t_implantation)*(t < t_implantation + t_rest) * 343
    baking = (t > t_implantation + t_rest)*(t < t_implantation + t_rest + t_baking)*350
    expression = implantation + rest + baking

    return expression


atom_density_W = 6.3e28  # atomic density m^-3
atom_density_Cu = 6.3e28  # atomic density m^-3
atom_density_CuCrZr = 6.3e28  # atomic density m^-3

id_W = 8
id_Cu = 7
id_CuCrZr = 6

id_top_surf = 9
id_coolant_surf = 10

t_implantation = 6000*400
t_rest = 6000*1800
t_baking = 30*24*3600

H = 29e-3
D3 = 16e-3

parameters = {
    "mesh_parameters": {
        "mesh_file": "maillage_monoblock/mesh_domains.xdmf",
        "cells_file": "maillage_monoblock/mesh_domains.xdmf",
        "facets_file": "maillage_monoblock/mesh_boundaries.xdmf",
        },
    "materials": [
        {
            # Tungsten
            "D_0": 2.9e-7,
            "E_diff": 0.39,
            "alpha": 1.29e-10,
            "beta": 6*atom_density_W,
            "thermal_cond": 120,
            "heat_capacity": 1,
            "rho": 2.89e6,
            "id": id_W,
        },
        {
            # Cu
            "D_0": 6.6e-7,
            "E_diff": 0.387,
            "alpha": 3.61e-10*atom_density_Cu**0.5,
            "beta": 1,
            "thermal_cond": 350,
            "heat_capacity": 1,
            "rho": 3.67e6,
            "id": id_Cu,
        },
        {
            # CuCrZr
            "D_0": 3.92e-7,
            "E_diff": 0.418,
            "alpha": 3.61e-10*atom_density_CuCrZr**0.5,
            "beta": 1,
            "thermal_cond": 350,
            "heat_capacity": 1,
            "rho": 3.67e6,
            "id": id_CuCrZr,
        },
        ],
    "traps": [
        {
            "density": 5e-4*atom_density_W,
            "energy": 1,
            "materials": [id_W]
        },
        # {
        #     "density": 5e-3*atom_density_W*(FESTIM.y > 0.014499),
        #     "energy": 1.4,
        #     "materials": [id_W]
        # },
        # {
        #     "density": 5e-5*atom_density_Cu,
        #     "energy": 0.5,
        #     "materials": [id_Cu]
        # },
        # {
        #     "density": 5e-5*atom_density_CuCrZr,
        #     "energy": 0.85,
        #     "materials": [id_CuCrZr]
        # },
        ],
    "boundary_conditions": [
        {
            "type": "dc",
            "surface": id_top_surf,
            "value": bc_top_H(t_implantation, t_rest, t_baking)
        },
        {
            "type": "recomb",
            "surface": id_coolant_surf,
            "Kr_0": 2.9e-14,
            "E_Kr": 1.92,
            "order": 2,
        },
        # {
        #     "type": "recomb",
        #     "surface": [id_left, id_right],
        #     "Kr_0": 3.2e-15,
        #     "E_Kr": 0.2
        # },
        ],
    "temperature": {
        "type": "solve_transient",
        "boundary_conditions": [
            {
                "type": "dirichlet",
                "value": bc_top_HT(t_implantation, t_rest, t_baking),
                "surface": id_top_surf
            },
            {
                "type": "dirichlet",
                "value": bc_coolant_HT(t_implantation, t_rest, t_baking),
                "surface": id_coolant_surf
            }
            ],
        "source_term": [
        ],
        "initial_condition": 273.15+200
        },
    "solving_parameters": {
        "final_time": t_implantation + t_rest + t_baking,
        "initial_stepsize": 10,
        "adaptive_stepsize": {
            "stepsize_change_ratio": 1.1,
            "t_stop": t_implantation + t_rest + t_baking + 1,
            "stepsize_stop_max": 200000,
            "dt_min": 1e-4,
            },
        "newton_solver": {
            "absolute_tolerance": 1e10,
            "relative_tolerance": 1e-9,
            "maximum_iterations": 10,
        }
        },
    "exports": {
        "xdmf": {
            # "functions": ['T', 'solute', '1', '2', '3', '4', 'retention'],
            # "labels":  ['temperature', 'solute', 'trap_1', 'trap_2',
            #             'trap_3', 'trap_4', 'retention'],
            "functions": ['T', 'solute', '1', 'retention'],
            "labels": ['T', 'solute', 'trap_1', 'retention'],
            "folder": 'Resultats/cas_test_simplifie_ITER'
        },
    }
}

FESTIM.generic_simulation.run(parameters, log_level=40)
