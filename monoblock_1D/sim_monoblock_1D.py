from context import FESTIM
from FESTIM.generic_simulation import run

# atom_density  =  density(g/m3)*Na(/mol)/M(g/mol)
atom_density_W = 6.3222e28  # atomic density m^-3
atom_density_Cu = 8.4912e28  # atomic density m^-3
atom_density_CuCrZr = 2.6096e28  # atomic density m^-3

# IDs for edges and surfaces (must be the same as in xdmf files)
id_W = 8
id_Cu = 7
id_CuCrZr = 6

id_top_surf = 1  #9
id_coolant_surf = 2  #10
# id_left_surf = 11

folder = "Solution_high_mesh_mu"


def rhoCp_W(T):
    return 5.15356e-6*T**3-8.30703e-2*T**2+5.98312e2*T+2.48160e6


def thermal_cond_W(T):
    return -7.84154e-9*T**3+5.03006e-5*T**2-1.07335e-1*T+1.75214e2


def rhoCp_Cu(T):
    return 1.68402e-4*T**3+6.14079e-2*T**2+4.67353e2*T+3.45899e6


def thermal_cond_Cu(T):
    return -7.84154e-9*T**3+5.03006e-5*T**2-1.07335e-1*T+1.75214e2


def rhoCp_CuCrZr(T):
    return -1.79134e-4*T**3+1.51383e-1*T**2+6.22091e2*T+3.46007e6


def thermal_cond_CuCrZr(T):
    return 5.25780e-7*T**3-6.45110e-4*T**2+2.57678e-01*T+3.12969e2

size = 8.5e-3
parameters = {
    "mesh_parameters": {
        "size": size,
        "initial_number_of_cells": 6000,
    },
    "materials": [
        {
            # Tungsten
            "D_0": 2.9e-7,
            "E_D": 0.39,
            "S_0": atom_density_W*1.3e-4,  # at/m3.Pa0.5 (from Grislia 2015)
            "E_S": 0.34,  # eV
            #"S_0": 1.0,  # case without solubility
            #"E_S": 0.0,  # case without solubility
            "thermal_cond": thermal_cond_W,
            "heat_capacity": 1,
            "rho": rhoCp_W,
            "borders": [0, 6e-3],
            "id": id_W,
        },
        {
            # Cu
            "D_0": 6.6e-7,
            "E_D": 0.387,
            "S_0": 3.12e28,  # at/m3.Pa0.5 (from ITER)
            "E_S": 0.572,  # eV
            #"S_0": 1.0,  # case without solubility
            #"E_S": 0.0,  # case without solubility
            "thermal_cond": thermal_cond_Cu,
            "heat_capacity": 1,
            "rho": rhoCp_Cu,
            "borders": [6e-3, 7e-3],
            "id": id_Cu,
        },
        {
            # CuCrZr
            "D_0": 3.92e-7,
            "E_D": 0.418,
            "S_0": 4.28e23,  # at/m3.Pa0.5 (from ITER)
            "E_S": 0.387,  # eV
            #"S_0": 1.0,  # case without solubility
            #"E_S": 0.0,  # case without solubility
            "thermal_cond": thermal_cond_CuCrZr,
            "heat_capacity": 1,
            "rho": rhoCp_CuCrZr,
            "borders": [7e-3, size],
            "id": id_CuCrZr,
        },
        ],
        "traps": [
        {
            "E_k": 0.39,
            "k_0": 2.9e-7/(1.1e-10**2)*0.8165/atom_density_W,
            "E_p": 1,
            "p_0": 8.4e12,
            "density": 5e-4*atom_density_W,
            "materials": [id_W]
        },
        # {
        #     "E_k": 0.39,
        #     "k_0": 2.9e-7/(1.1e-10**2)*0.8165/atom_density_W,
        #     "E_p": 1.4,
        #     "p_0": 8.4e12,
        #     "density": 5e-3*atom_density_W*(FESTIM.y > 0.014499),
        #     "materials": [id_W]
        # },
        {
            "E_k": 0.387,
            "k_0": 6.6e-7/(3.61e-10**2)/atom_density_Cu,
            "E_p": 0.5,
            "p_0": 7.98e13,
            "density": 5e-5*atom_density_Cu,
            "materials": [id_Cu]
        },
        {
            "E_k": 0.418,
            "k_0": 3.92e-7/(3.61e-10**2)/atom_density_CuCrZr,
            "E_p": 0.85,
            "p_0": 7.98e13,
            "density": 5e-5*atom_density_CuCrZr,
            "materials": [id_CuCrZr]
        },
        ],
    "boundary_conditions": [
        {
            "type": "dc",
            "surfaces": id_top_surf,
            "value": 5e22*(FESTIM.t>20) + (5e22*(FESTIM.t/20))*(FESTIM.t<20)
        },
        {
            "type": "recomb",
            "surfaces": id_coolant_surf,
            "Kr_0": 2.9e-14,
            "E_Kr": 1.92,
            "order": 2,
        },
        # {
        #     "type": "dc",
        #     "surfaces": id_coolant_surf,
        #     "value": 0
        # },
        # {
        #     "type": "recomb",
        #     "surfaces": [id_left_surf],
        #     "Kr_0": 2.9e-18,
        #     "E_Kr": 1.16,
        #     "order": 2,
        # },
        ],
    "temperature": {
        "type": "solve_stationary",
        "boundary_conditions": [
            {
                "type": "dc",
                "value": 1200,
                "surfaces": id_top_surf
            },
            {
                "type": "dc",
                "value": 370,
                "surfaces": id_coolant_surf
            }
            ],
        "source_term": [
        ],
        "initial_condition": 273.15+200
        },
    "solving_parameters": {
        # "type": "solve_stationary",
        "final_time": 1e7,
        "initial_stepsize": 100000,
        "adaptive_stepsize": {
            "stepsize_change_ratio": 1.05,
            "t_stop": 1e8,
            "stepsize_stop_max": 1e7,
            "dt_min": 1e-8,
            },
        "times": [],
        "newton_solver": {
            "absolute_tolerance": 1e11,
            "relative_tolerance": 1e-10,
            "maximum_iterations": 10,
        }
        },
    "exports": {
        "xdmf": {
            "functions": ['T', '0', '1', '2', '3', 'retention'],
            "labels": ['T', 'solute', '1', '2', '3', 'retention'],
            "folder": folder,
            "all_timesteps": True,
        },
        "derived_quantities": {
            "total_volume": [
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "solute"
                },
                {
                    "volumes": [id_W],
                    "field": "1"
                },
                {
                    "volumes": [id_Cu],
                    "field": "2"
                },
                {
                    "volumes": [id_CuCrZr],
                    "field": "3"
                },
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "retention"
                },
            ],
            "surface_flux": [
                {
                    "surfaces": [id_top_surf, id_coolant_surf],
                    "field": "solute"
                }
            ],
            "file": "derived_quantities.csv",
            "folder": folder
        }
    }

}
run(parameters, log_level=30)
