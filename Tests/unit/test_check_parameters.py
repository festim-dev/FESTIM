import FESTIM
import fenics
import pytest


def test_materials_attribute():
    """
    Checks that the materials list is passed correctly to Simulation()
    """
    materials = [
        {
            "D_0": 1,
            "E_D": 2,
            "S_0": 3,
            "E_S": 4,
            "id": 1,
            "rho": 5,
            "heat_capacity": 6,
            "thermal_cond": 7,
            "borders": [0, 1],
        },
        {
            "D_0": 2,
            "E_D": 3,
            "S_0": 4,
            "E_S": 5,
            "id": 2,
            "rho": 6,
            "heat_capacity": 7,
            "thermal_cond": 8,
            "borders": [1, 2],
        },
    ]
    my_sim = FESTIM.Simulation(
        {   "boundary_conditions": [],
            "materials": materials,
         "temperature": {"type": "solve_transient"}})
    for material, sim_mat in zip(materials, my_sim.materials):
        for key, value in material.items():
            assert getattr(sim_mat, key) == value


# def test_keys_dont_match():
#     """
#     Checks that Errors are raised when the materials keys don't match
#     """
#     materials = [
#         {
#             "D_0": 1,
#             "E_D": 2,
#             "S_0": 3,
#             "E_S": 4,
#             "id": 1,
#             "heat_capacity": 6,
#             "thermal_cond": 7,
#             "borders": [0, 1],
#         },
#         {
#             "D_0": 2,
#             "E_D": 3,
#             "S_0": 4,
#             "E_S": 5,
#             "id": 2,
#             "rho": 6,
#             "heat_capacity": 7,
#             "thermal_cond": 8,
#             "borders": [1, 2],
#         },
#     ]
#     my_sim = FESTIM.Simulation(
#         {"boundary_conditions": [],
#          "materials": materials,
#          "temperature": {"type": "solve_transient"}})
#     with pytest.raises(ValueError, match=r"keys are not the same"):
#         my_sim.check_materials()

#     materials = [
#         {
#             "D_0": 1,
#             "E_D": 2,
#             "S_0": 3,
#             "E_S": 4,
#             "id": 1,
#             "rho": 6,
#             "heat_capacity": 6,
#             "thermal_cond": 7,
#             "borders": [0, 1],
#         },
#         {
#             "D_0": 2,
#             "E_D": 3,
#             "S_0": 4,
#             "E_S": 5,
#             "id": 2,
#             "heat_capacity": 7,
#             "thermal_cond": 8,
#             "borders": [1, 2],
#         },
#     ]
#     my_sim = FESTIM.Simulation(
#         {"boundary_conditions": [],
#          "materials": materials,
#          "temperature": {"type": "solve_transient"}})
#     with pytest.raises(ValueError, match=r"keys are not the same"):
#         my_sim.check_materials()


# def test_unknown_keys():
#     """
#     Checks warning when there's an unknown material key
#     """
#     materials = [
#         {
#             "D_0": 1,
#             "E_D": 2,
#             "S_0": 3,
#             "E_S": 4,
#             "id": 1,
#             "coucou": 2,
#             "heat_capacity": 6,
#             "thermal_cond": 7,
#             "borders": [0, 1],
#         },
#     ]
#     my_sim = FESTIM.Simulation(
#         {"boundary_conditions": [],
#          "materials": materials,
#          "temperature": {"type": "solve_transient"}})
#     with pytest.warns(
#             UserWarning, match=r"coucou key in materials is unknown"):
#         my_sim.check_materials()


def test_unused_keys():
    """
    Checks warnings when some keys are unused
    """
    materials = [
        {
            "D_0": 1,
            "E_D": 2,
            "S_0": 3,
            "E_S": 4,
            "id": 1,
            "rho": 1,
            "borders": [0, 1],
        },
    ]
    my_sim = FESTIM.Simulation(
        {"boundary_conditions": [],
         "materials": materials,
         "temperature": {"type": "expression"},
         })
    with pytest.warns(
            UserWarning, match=r"rho key will be ignored"):
        my_sim.check_materials()


    materials[0].pop('rho', None)
    materials[0]["heat_capacity"] = 2
    my_sim = FESTIM.Simulation(
        {"boundary_conditions": [],
         "materials": materials,
         "temperature": {"type": "expression"},
         })
    with pytest.warns(
            UserWarning, match=r"heat_capacity key will be ignored"):
        my_sim.check_materials()


def test_unused_thermal_cond():
    """
    Checks warnings when some keys are unused
    """
    materials = [
        {
            "D_0": 1,
            "E_D": 2,
            "S_0": 3,
            "E_S": 4,
            "thermal_cond": 2,
            "id": 1,
            "borders": [0, 1],
        },
    ]
    my_sim = FESTIM.Simulation(
        {"boundary_conditions": [],
         "materials": materials,
         })
    my_sim.parameters["temperature"] = {"type": "expression"}
    my_sim.parameters["exports"] = {}
    with pytest.warns(
            UserWarning, match=r"thermal_cond key will be ignored"):
        my_sim.check_materials()

    # this shouldn't throw warnings
    my_sim.parameters["exports"] = {
        "derived_quantities": {
            "surface_flux": [
                {
                    "field": "T",
                    "surfaces": [0, 1]
                }
                ]
            }
    }
    # record the warnings
    with pytest.warns(None) as record:
        my_sim.check_materials()

    # check that no warning were raised
    assert len(record) == 0


def test_different_ids_in_materials():
    """
    Checks that an error is raised when two materials have the same id
    """
    materials = [
        {
            "D_0": 1,
            "E_D": 2,
            "id": 1,
            "borders": [0, 1],
        },
        {
            "D_0": 2,
            "E_D": 3,
            "id": 1,
            "borders": [1, 2],
        },
    ]
    my_sim = FESTIM.Simulation(
        {"boundary_conditions": [],
         "materials": materials,
         "temperature": {"type": "expression"}})
    with pytest.raises(ValueError, match=r"Some materials have the same id"):
        my_sim.check_materials()
