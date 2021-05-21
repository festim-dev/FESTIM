def find_material_from_id(materials, id):
    '''Returns the material from a given id
    Parameters:
    - materials : list of dicts ex: [{"id": 2}, {"id": 3}, {"id": 5}]
    - id : int
    '''
    for material in materials:
        if material['id'] == id:
            return material
    raise ValueError("Couldn't find ID " + str(id) + " in materials list")
    return


def update_expressions(expressions, t):
    '''Update all FEniCS Expression() in expressions.

    Arguments:
    - expressions: list, contains the fenics Expression
    to be updated.
    - t: float, time.
    '''
    for expression in expressions:
        expression.t = t
    return expressions


bc_types = {
    "dc": ["dc", "solubility", "dc_imp"],
    "neumann": ["flux"],
    "robin": ["recomb"]
}

quantity_types = [
    "surface_flux",
    "average_volume",
    "average_surface",
    "maximum_volume",
    "minimum_volume",
    "total_volume",
    "total_surface"
    ]

field_types = [
    "solute",
    "retention",
    "T"
]
parameters_helper = {
    "materials": {
        "H": {
            "description": "[insert description]",
            "unit": "[insert unit]"
        },
        "E_D": {
            "description": "Diffusion coefficient activation energy",
            "unit": "eV"
            },
        "D_0": {
            "description": "Diffusion coefficient pre exponential factor",
            "unit": "(m^2/s)"
            },
        "S_0": {
            "description": "Solubility coefficient pre exponential factor",
            "unit": "(m^2/s)"
            },
        "E_S": {
            "description": "Solubility coefficient activation energy",
            "unit": "eV"
            },
        "borders": {
            "description": "1D only: delimitations of the domain. \
            Exemple: [0, 0.5]",
            "unit": "m"
            },
        "thermal_cond": {
            "description": "required if heat equation is solved or if thermal\
                 flux is computed. thermal conductivity",
            "unit": "W/m/K"
            },
        "heat_capacity": {
            "description": "required if heat equation is solved. \
                Heat capacity",
            "unit": "J/K/kg"
            },
        "rho": {
            "description": "required if heat equation is solved. \
                Density",
            "unit": "kg/m^3"
            },
        "id": {
            "description": "id of the domain. If a marked mesh is given,\
                id must correspond to the volume_markers_file",
            "unit": "None"
            },
    },
    "traps": {
        "E_k": {
            "description": "Trapping rate activation energy",
            "unit": "eV"
            },
        "k_0": {
            "description": "Trapping rate pre-exponential factor",
            "unit": "m^3/s"
            },
        "E_p": {
            "description": "Detrapping rate activation energy",
            "unit": "eV"
            },
        "p_0": {
            "description": "Detrapping rate pre-exponential factor",
            "unit": "s^-1"
            },
        "density": {
            "description": "denisty of the trap. Can be float or an expression\
                 (ex: (1 + FESTIM.x)*FESTIM.t<100)",
            "unit": "m^-3"
            },
        "materials": {
            "description": "ids of the domains where the trap is present.\
                 Can be int or list of int",
            "unit": "None"
            },
        "source_term": {
            "description": "Volumetric source term for the trapped population",
            "unit": "m^-3.s^-1"
            },
        },
    "boundary_conditions": {
        "type": {
            "dc": "Dirichlet boundary condition",
            "flux": "Pure Neumann boundary condition",
            "recomb": "Recombination flux",
            "convective_flux": "Convective exchange (for heat transfer)",
            "solubility": "Dirichlet boundary condition based on solubility and pressure c=S*P^0.5",
            "table": "Dirichlet boundary condition based on interpolated values from 2D table (t, c(t)).",
            "dc_imp": "Dirichlet boundary condition based on triangular model for volumetric implantation"
            },
        "surfaces": {
            "description": "List of surfaces on which the boundary condition is applied",
            "unit": "None"
            },
        "value": {
                    "description": "Value of boundary conditions only needed for types dc and flux. Can be float or an expression (ex: (1 + FESTIM.x)*FESTIM.t<100)",
                    "unit": "m^-3 or m^-2.s^-1"
                },
        "component": {
                    "description": "int in [0;N], N being the number of traps. By default 0, the solute population",
                    "unit": "None"
                },
        "Kr_0": {
                    "description": "Value of recombination coefficient pre-exponential factor",
                    "unit": "m^(-2+3*n).s^-1 , where n is the order of recombination"
                },
        "E_Kr": {
                    "description": "Value of recombination coefficient activation energy",
                    "unit": "eV"
                },
        "S_0": {
                    "description": "Value of solubility coefficient pre-exponential factor",
                    "unit": "m^-3.Pa^-0.5"
                },
        "E_S": {
                    "description": "Value of solubility coefficient activation energy",
                    "unit": "eV"
                },
        "pressure": {
                    "description": "Value of pressure",
                    "unit": "Pa"
        }
    },
    "mesh_parameters": {
        "refinements": {},
        "size": {
            "description": "Size of the 1D problem",
            "unit": "m"
        },
        "initial_number_of_cells": {
            "description": "Number of cells in the domain before refinement",
            "unit": "cells"
        },
        "mesh_file": {},
        "cells_file": {},
        "facets_file": {},
        "meshfunction_cells": {},
        "meshfunction_facets": {},
    },
    "temperature": {},
    "solving_parameters": {},
    "exports": {},
    }


def help_key(key):
    def find_dict(key, var, res=[]):
        if key in var:
            res.append(var[key])
        else:
            for k in var.keys():
                if type(var[k]) == dict:
                    find_dict(key, var=var[k], res=res)
        return res
    d = find_dict(key, parameters_helper)
    for k in d:
        print(k)
