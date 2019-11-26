from context import FESTIM
import sympy as sp


# Definition des BCs
def bc_top_H(t_implantation, t_rest, t_baking):
    t = FESTIM.t
    implantation = (t < t_implantation) * \
        1e23*2.5e-9/(2.9e-7*sp.exp(-0.39/FESTIM.k_B/1200)) / \
        (1.3e-4 * atom_density_W * sp.exp(-0.34/FESTIM.k_B/1200))  # Solubility W
    expression = implantation

    return expression


def bc_top_HT(t_implantation, t_rest, t_baking):
    t = FESTIM.t
    implantation = (t < t_implantation) * 1200
    rest = (t > t_implantation)*(t < t_implantation + t_rest) * 343
    baking = (t > t_implantation + t_rest)*(350+273.15)
    expression = implantation + rest + baking
    return expression


def bc_coolant_HT(t_implantation, t_rest, t_baking):
    t = FESTIM.t
    implantation = (t < t_implantation) * 373
    rest = (t > t_implantation)*(t < t_implantation + t_rest) * 343
    baking = (t > t_implantation + t_rest)*(350+273.15)
    expression = implantation + rest + baking

    return expression


# Definition des paramètres
# atom_density  =  density(g/m3)*Na(/mol)/M(g/mol)
atom_density_W = 6.3222e28  # atomic density m^-3
atom_density_Cu = 8.4912e28  # atomic density m^-3
atom_density_CuCrZr = 2.6096e28  # atomic density m^-3

# Definition des id (doit etre les memes que dans le maillage xdmf)
id_W = 8
id_Cu = 7
id_CuCrZr = 6

id_top_surf = 9
id_coolant_surf = 10
id_left_surf = 11

# Definition des temps
t_implantation = 6000*400
t_rest = 47696400-t_implantation
t_baking = 50648400-t_rest

# Definition du fichier de stockage
folder = 'results/ITER_case_theta/'

# Dict parameters
parameters = {
    "mesh_parameters": {
        "mesh_file": "maillages/Mesh_ITER/mesh_domains.xdmf",
        "cells_file": "maillages/Mesh_ITER/mesh_domains.xdmf",
        "facets_file": "maillages/Mesh_ITER/mesh_boundaries.xdmf",
        },
    "materials": [
        {
            # Tungsten
            "D_0": 2.9e-7,
            "E_diff": 0.39,
            "S_0": atom_density_W*1.3e-4,  # at/m3.Pa0.5 (from Grislia 2015)
            "E_S": 0.34,  # eV
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
            "E_diff": 0.387,
            "S_0": 3.12e28,  # at/m3.Pa0.5 (from ITER)
            "E_S": 0.572,  # eV
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
            "E_diff": 0.418,
            "S_0": 4.28e23,  # at/m3.Pa0.5 (from ITER)
            "E_S": 0.387,  # eV
            "alpha": 3.61e-10*(atom_density_CuCrZr)**0.5,
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
        {
            "density": 5e-3*atom_density_W*(FESTIM.y > 0.014499),
            "energy": 1.4,
            "materials": [id_W]
        },
        {
            "density": 5e-5*atom_density_Cu,
            "energy": 0.5,
            "materials": [id_Cu]
        },
        {
            "density": 5e-5*atom_density_CuCrZr,
            "energy": 0.85,
            "materials": [id_CuCrZr]
        },
        ],
    "boundary_conditions": [
        {
            "type": "dc",
            "surface": id_top_surf,
            "value": bc_top_H(t_implantation, t_rest, t_baking)
        },
        {
            "type": "dc",
            "surface": id_left_surf,
            "value": 0
        },
        {
            "type": "recomb",
            "surface": id_coolant_surf,
            "Kr_0": 2.9e-14,
            "E_Kr": 1.92,
            "order": 2,
        },
        # {
        #     "type": "dc",
        #     "surface": id_left_surf,
        #     "value": 0
        # },
        # {
        #     "type": "recomb",
        #     "surface": [id_left_surf],
        #     "Kr_0": 2.9e-18,
        #     "E_Kr": 1.16,
        #     "order": 2,
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
        "initial_stepsize": 1,
        "adaptive_stepsize": {
            "stepsize_change_ratio": 1.1,
            "t_stop": t_implantation + t_rest*8/10,
            "stepsize_stop_max": t_baking/15,
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
            "functions": ['T', '0', '1', '2', '3', '4', 'retention'],
            "labels": ['T', '0', '1', '2', '3', '4', 'sum'],
            "folder": folder
        },
        "derived_quantities": {
            "total_volume": [
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "solute"
                },
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "1"
                },
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "2"
                },
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "3"
                },
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "4"
                },
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "retention"
                },
            ],
            "file": "derived_quantities.csv",
            "folder": folder
        }
    }
}
