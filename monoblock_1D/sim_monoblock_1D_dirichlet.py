from context import FESTIM
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

folder = "Solution_high_mesh_mu"


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
    return -3.93153e-08*T**3+3.76147e-05*T**2-7.88669e-02*T+4.02301e+02
   # return -7.84154e-9*T**3+5.03006e-5*T**2-1.07335e-1*T+1.75214e2


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
        "initial_number_of_cells": 10000,
        # "refinements": [
        #     {
        #         "cells": 500,
        #         "x": 3e-6,
        #     },
        #     {
        #         "cells": 100,
        #         "x": 30e-9,
        #     },
        #     ]
        },
    "materials": [
        {
            # Tungsten
            "D_0": 2.9e-7*0.8165,  # OK TMAP eq(6)
            "E_D": 0.39,  # OK TMAP eq(6)
            "S_0": 1.87e24,  # OK MUYI mais attention units ? at/m3.Pa0.5
            "E_S": 1.04,  # OK MUYI (eV)
            #"S_0": 1.0,  # case without solubility
            #"E_S": 0.0,  # case without solubility
            "thermal_cond": thermal_cond_W,
            "heat_capacity": 1,
            "rho": rhoCp_W,
            "borders": [0, 6e-3],  # OK TMAP
            "id": id_W,
        },
        {
            # Cu
            "D_0": 6.6e-7,  # OK TMAP eq(9)
            "E_D": 0.387,  # OK TMAP eq(9)
            "S_0": 3.14e24,  # OK TMAP eq(8) at/m3.Pa0.5
            "E_S": 0.572,  # OK TMAP eq(8) eV
            #"S_0": 1.0,  # case without solubility
            #"E_S": 0.0,  # case without solubility
            "thermal_cond": thermal_cond_Cu,
            "heat_capacity": 1,
            "rho": rhoCp_Cu,
            "borders": [6e-3, 7e-3],  # OK TMAP
            "id": id_Cu,
        },
        {
            # CuCrZr
            "D_0": 3.92e-7,  # OK TMAP eq(13)
            "E_D": 0.418,  # OK TMAP eq(13)
            "S_0": 4.28e23,  # OK TMAP eq(12) at/m3.Pa0.5 (from ITER)
            "E_S": 0.387,  # OK TMAP eq(12) eV
            #"S_0": 1.0,  # case without solubility
            #"E_S": 0.0,  # case without solubility
            "thermal_cond": thermal_cond_CuCrZr,
            "heat_capacity": 1,
            "rho": rhoCp_CuCrZr,
            "borders": [7e-3, size],  # OK TMAP
            "id": id_CuCrZr,
        },
        ],
    "traps": [
        {
            "E_k": 0.39,  # OK TMAP eq (3)
            "k_0": 2.9e12*0.8165/atom_density_W,  # OK TMAP eq (3)
            #"k_0": 2.9e-7/(1.1e-10**2)*0.8165/atom_density_W,
            "E_p": 1.2,  # OK TMAP eq (4)
            "p_0": 8.4e12,  # OK TMAP eq (4)
            "density": 5e-4 * atom_density_W,  # OK TMAP ligne (72)
            "materials": [id_W]
        },
        {
             "E_k": 0.39,  # OK TMAP eq (3)
             "k_0": 2.9e12*0.8165/atom_density_W,  # OK TMAP eq (3)
             "E_p": 1.4,  # OK TMAP eq (5)
             "p_0": 8.4e12,  # OK TMAP eq (5)
             "density": 5e-3*atom_density_W*(FESTIM.x < 1e-6),  # OK TMAP ligne (75)
             "materials": [id_W]
         },
        {
            "E_k": 0.387,  # OK TMAP eq (10)
            "k_0": 6.6e-7/(3.61e-10**2)/atom_density_Cu,  # OK TMAP eq (10)
            "E_p": 0.5,  # OK TMAP eq (11)
            "p_0": 7.98e13,  # OK TMAP eq (11)
            "density": 5e-5*atom_density_Cu,  # OK TMAP ligne (89)
            "materials": [id_Cu]
        },
        {
            "E_k": 0.418,  # OK TMAP eq (15)
            "k_0": 3.92e-7/(3.61e-10**2)/atom_density_CuCrZr,  # OK TMAP eq (15)
            "E_p": 0.5,  # OK TMAP eq (11)
            "p_0": 7.98e13,  # OK TMAP eq (11)
            "density": 5e-5*atom_density_CuCrZr,  # OK TMAP ligne (103)
            "materials": [id_CuCrZr]
        },
        {
            "E_k": 0.418,  # OK TMAP eq (15)
            "k_0": 3.92e-7/(3.61e-10**2)/atom_density_CuCrZr,  # OK TMAP eq (15)
            "E_p": 0.83,  # OK TMAP eq (16)
            "p_0": 7.98e13,  # OK TMAP eq (16)
            "density": 0.04*atom_density_CuCrZr,  # OK TMAP ligne (104)
            "materials": [id_CuCrZr]
        },
        ],
    # "source_term": [
    #    {
    #         'value': 5e23 * 1/2.5e-9 * (FESTIM.x < 2.5e-9),  #* (FESTIM.t <= 400)
    #         'volumes': [id_W]
    #    },
    #    ],    
    "boundary_conditions": [
        {
           "type": "dc",
           "surfaces": id_top_surf,
           #"value": 5e23 (flux) + recombinaison (kr0=3.2e-15 : Ekr=-1.16 ) + pression (ligne 168)
           "value" : 1.223634016511513351e+23

        #    "type": "recomb",
        #    "surfaces": id_top_surf,
        #     "Kr_0": 3.2e-15,
        #     "E_Kr": 1.16,
        #     "order": 2,
        },
        # {
        #     "type": "dc",
        #     "value": 0,
        #     "surfaces": id_top_surf
        # },
        {
            "type": "recomb",
            "surfaces": id_coolant_surf,
            "Kr_0": 2.9e-14, # OK TMAP eq(13)
            "E_Kr": 1.92, # OK TMAP eq(14)
            "order": 2,
        },
        # {
        #     "type": "dc",
        #     "surfaces": id_coolant_surf,
        #     "value": 0
        # },
        ],
    "temperature": {
        "type": "solve_stationary",
        "boundary_conditions": [
            {
                "type": "dc",
                "value": 1200, # OK TMAP phase 1 tab(3) ligne 172/173
                "surfaces": id_top_surf
            },
            {
                "type": "dc",
                "value": 373, # OK TMAP phase 1 tab(3) ligne 182/183
                "surfaces": id_coolant_surf
            }
            ],
        "source_term": [
        ],
        # en toute rigueur il faudrait une rampe tempd ligne 37 ; 47 ; 57
        #"initial_condition": 273.15+200
        },
    "solving_parameters": {
        # "type": "solve_stationary",
        "final_time": 2400000, # OK TMAP ligne 183
        "initial_stepsize": 1000,
        "adaptive_stepsize": {
            "stepsize_change_ratio": 1.2,
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
        "txt": {
            "times": [2400000],
            "functions": ['T','0', '1', '2', '3', 'retention'],
            "labels": ['T','0','1', '2', '3', 'retention'],
            "folder": folder,
        },
         "xdmf": {
             "functions": ['T', '0',  'retention'],
             "labels": ['T', '0', 'retention'],
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
                    "volumes": [id_W],
                    "field": "2"
                },
                {
                    "volumes": [id_Cu],
                    "field": "3"
                },
                {
                    "volumes": [id_CuCrZr],
                    "field": "4"
                },
                {
                    "volumes": [id_CuCrZr],
                    "field": "5"
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
