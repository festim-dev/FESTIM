from context import FESTIM
import sympy as sp


# BCs definition
def bc_top_H(t_implantation, t_rest, t_baking):
    t = FESTIM.t
    implantation = (t <= t_implantation) * \
       1e23*2.5e-9/(2.9e-7*sp.exp(-0.39/FESTIM.k_B/1200)) # S=1.0
    expression = implantation

    return expression


def bc_top_HT(t_implantation, t_rest, t_baking):
    t = FESTIM.t
    implantation = (t <= t_implantation) * 1200
    rest = (t > t_implantation)*(t <= (t_implantation + t_rest) ) * 343
    baking = (t > (t_implantation + t_rest) )*(350+273.15)
    expression = implantation + rest + baking
    return expression


def bc_coolant_HT(t_implantation, t_rest, t_baking):
    t = FESTIM.t
    implantation = (t <= t_implantation) * 373
    rest = (t > t_implantation)*(t <= t_implantation + t_rest) * 343
    baking = (t > t_implantation + t_rest)*(350+273.15)
    expression = implantation + rest + baking

    return expression


# Parameters
# atom_density  =  density(g/m3)*Na(/mol)/M(g/mol)
atom_density_W = 6.3222e28  # atomic density m^-3
atom_density_Cu = 8.4912e28  # atomic density m^-3
atom_density_CuCrZr = 2.6096e28  # atomic density m^-3

# IDs for edges and surfaces (must be the same as in xdmf files)
id_W = 8
id_Cu = 7
id_CuCrZr = 6

id_top_surf = 9
id_coolant_surf = 10
id_left_surf = 11

# Times
t_implantation = 6000*400
t_rest = 47696400-t_implantation
t_baking = 50648400-t_rest-t_implantation

# Storage folder
folder = 'results/temp/'

# Dict parameters
parameters = {
    "mesh_parameters": { # à vérifier pour que ça fonctionne chez toi
        "mesh_file": "maillages/Mesh_ITER_58734/mesh_domains_58734.xdmf",
        "cells_file": "maillages/Mesh_ITER_58734/mesh_domains_58734.xdmf",
        "facets_file": "maillages/Mesh_ITER_58734/mesh_lines_58734.xdmf",
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
            "alpha": (2.9e-7*atom_density_W/(2.9e12*1.0000))**0.5,  # 1.0000 coef H/D/T
            "beta": 1,
            "thermal_cond": 120,
            "heat_capacity": 1,
            "rho": 2.89e6,
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
            "alpha": 3.61e-10*(atom_density_Cu)**0.5,
            "beta": 1,
            "thermal_cond": 350,
            "heat_capacity": 1,
            "rho": 3.67e6,
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
            "alpha": 3.61e-10*(atom_density_CuCrZr)**0.5,
            "beta": 1,
            "thermal_cond": 350,
            "heat_capacity": 1,
            "rho": 3.67e6,
            "id": id_CuCrZr,
        },
        ],
    "traps": [
        # {
        #     "E_k": 0.39, # OK TMAP eq (3)
        #     "k_0": 2.9e12*0.8165/atom_density_W, # OK TMAP eq (3)
        #     #"k_0": 2.9e-7/(1.1e-10**2)*0.8165/atom_density_W, 
        #     "E_p": 1.2, # OK TMAP eq (4)
        #     "p_0": 8.4e12, # OK TMAP eq (4)
        #     "density": 5e-4*atom_density_W, # OK TMAP ligne (72)
        #     "materials": [id_W]
        # },
        # {
        #      "E_k": 0.39, # OK TMAP eq (3)
        #      "k_0": 2.9e12*0.8165/atom_density_W, # OK TMAP eq (3)
        #      "E_p": 1.4, # OK TMAP eq (5)
        #      "p_0": 8.4e12, # OK TMAP eq (5)
        #      "density": 5e-3*atom_density_W*(FESTIM.y < 1e-6), # OK TMAP ligne (75)
        #      "materials": [id_W]
        #  },
        # {
        #     "E_k": 0.387, # OK TMAP eq (10)
        #     "k_0": 6.6e-7/(3.61e-10**2)/atom_density_Cu, # OK TMAP eq (10)
        #     "E_p": 0.5, # OK TMAP eq (11)
        #     "p_0": 7.98e13, # OK TMAP eq (11)
        #     "density": 5e-5*atom_density_Cu, # OK TMAP ligne (89)
        #     "materials": [id_Cu]
        # },
        # {
        #     "E_k": 0.418, # OK TMAP eq (15)
        #     "k_0": 3.92e-7/(3.61e-10**2)/atom_density_CuCrZr,  # OK TMAP eq (15)
        #     "E_p": 0.5,  # OK TMAP eq (11)
        #     "p_0": 7.98e13,  # OK TMAP eq (11)
        #     "density": 5e-5*atom_density_CuCrZr, # OK TMAP ligne (103)
        #     "materials": [id_CuCrZr]
        # },
        #         {
        #     "E_k": 0.418, # OK TMAP eq (15)
        #     "k_0": 3.92e-7/(3.61e-10**2)/atom_density_CuCrZr,  # OK TMAP eq (15)
        #     "E_p": 0.83,  # OK TMAP eq (16)
        #     "p_0": 7.98e13,  # OK TMAP eq (16)
        #     "density": 0.04*atom_density_CuCrZr, # OK TMAP ligne (104)
        #     "materials": [id_CuCrZr]
        # },
        ],
    "boundary_conditions": [
        {
            "type": "dc",
            "surfaces": id_top_surf,
            "value": bc_top_H(t_implantation, t_rest, t_baking)
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
        #     "surfaces": id_left_surf,
        #     "value": 0
        # },
        {
            "type": "recomb",
            "surfaces": [id_left_surf],
            "Kr_0": 2.9e-18,
            "E_Kr": 1.16,
            "order": 2,
        },
        ],
    "temperature": {
        "type": "solve_transient",
        "boundary_conditions": [
            {
                "type": "dc",
                "value": bc_top_HT(t_implantation, t_rest, t_baking),
                "surfaces": id_top_surf
            },
            {
                "type": "dc",
                "value": bc_coolant_HT(t_implantation, t_rest, t_baking),
                "surfaces": id_coolant_surf
            }
            ],
        "source_term": [
        ],
        "initial_condition": 273.15+200
        },
    "solving_parameters": {
        "final_time": t_implantation + t_rest + t_baking,
        "times": [t_implantation,
                  t_implantation+t_rest,
                  t_implantation+t_rest+t_baking],
        "initial_stepsize": 100000,
        "adaptive_stepsize": {
            "stepsize_change_ratio": 1.3,
            "t_stop": t_implantation + t_rest,
            "stepsize_stop_max": t_baking/10,
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
            "functions": ['T', '0', 'retention'],
            # "functions": ['T', '0', '1', '2', '3', '4', 'retention'],
            # "labels": ['T', 'solute', '1', '2', '3', '4', 'retention'],
            "labels": ['T', 'solute', 'retention'],
            "folder": folder,
            "all_timesteps": True,
        },
        # "derived_quantities": {
        #     "total_volume": [
        #         {
        #             "volumes": [id_W, id_Cu, id_CuCrZr],
        #             "field": "solute"
        #         },
        #         {
        #             "volumes": [id_W],
        #             "field": "1"
        #         },
        #         {
        #             "volumes": [id_W],
        #             "field": "2"
        #         },
        #         {
        #             "volumes": [id_Cu],
        #             "field": "3"
        #         },
        #         {
        #             "volumes": [id_CuCrZr],
        #             "field": "4"
        #         },
        #         {
        #             "volumes": [id_W, id_Cu, id_CuCrZr],
        #             "field": "retention"
        #         },
        #     ],
        #     "surface_flux": [
        #         {
        #             "surfaces": [id_coolant_surf, id_left_surf],
        #             "field": "solute"
        #         }
        #     ],
        #     "file": "derived_quantities.csv",
        #     "folder": folder
        # }
    }
}
