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
