import FESTIM


def test_source_terms_as_list():
    """Simply checks parameters dict is correctly read
    """
    parameters = {
        "materials": [
            {
                "E_D": 1,
                "D_0": 1,
                "id": 1
                }
                ],
        "traps": [
            {
                "k_0": 1,
                "E_k": 0,
                "p_0": 1,
                "E_p": 0,
                "materials": [1],
                "density": 0
            }
            ],
        "initial_conditions": [
        ],
        "source_term": [
            {
                "value": 1,
                "volume": 1
            },
            {
                "value": 1,
                "volume": [1]
            }
        ],
        "mesh_parameters": {
                "initial_number_of_cells": 10,
                "size": 1,
            },
        "boundary_conditions": [
            {
                "type": "dc",
                "value": 0,
                "surfaces": [1, 2]
            }
            ],
        "temperature": {
                'type': "expression",
                'value': 30
            },
        "solving_parameters": {
            "type": "solve_stationary",
            "newton_solver": {
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 2,
            },
        },
        "exports": {
        },
    }
    FESTIM.run(parameters)


def test_read_fluxes():
    """Tests that fluxes can be read from parameters
    """
    parameters = {
        "materials": [
            {
                "E_D": 1,
                "D_0": 1,
                "thermal_cond": 2,
                "id": 1
                }
                ],
        "traps": [],
        "initial_conditions": [],
        "source_term": [],
        "mesh_parameters": {
                "initial_number_of_cells": 10,
                "size": 1,
            },
        "boundary_conditions": [
            {
                "type": "flux",
                "value": 0,
                "surfaces": [1, 2]
            },
            {
                "surfaces": [2],
                "type": "recomb",
                "Kr_0": 1e-9,
                "E_Kr": 1,
                "order": 2,
            }
            ],
        "temperature": {
                'type': "solve_stationary",
                'boundary_conditions': [
                    {
                        "type": "flux",
                        "value": 1,
                        "surfaces": [2]
                    },
                    {
                        "type": "convective_flux",
                        "h_coeff": 1,
                        "T_ext": 2,
                        "surfaces": [1]
                    },
                ]
            },
        "solving_parameters": {
            "type": "solve_stationary",
            "newton_solver": {
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 2,
            },
        },
        "exports": {},
    }
    my_sim = FESTIM.Simulation(parameters)
    print(my_sim.boundary_conditions[1].prms)
    my_sim.initialise()
