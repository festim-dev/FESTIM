from FESTIM import Material, Materials
import pytest


def test_find_material_from_id():
    """Tests the function find_material_from_id() for cases with one id per
    material
    """
    mat_1 = Material(id=1, D_0=None, E_D=None)
    mat_2 = Material(id=2, D_0=None, E_D=None)
    my_Mats = Materials([mat_1, mat_2])
    assert my_Mats.find_material_from_id(1) == mat_1
    assert my_Mats.find_material_from_id(2) == mat_2


def test_find_material_from_id_with_several_ids():
    """Tests the function find_material_from_id() for cases with several ids
    per material
    """

    mat_1 = Material(id=[1, 2], D_0=None, E_D=None)
    my_Mats = Materials([mat_1])
    assert my_Mats.find_material_from_id(1) == mat_1
    assert my_Mats.find_material_from_id(2) == mat_1


def test_find_material_from_id_unfound_id():
    """
    Tests the function find_material_from_id with a list of materials
    without the searched ID
        - check that an error is rasied
    """
    mat_1 = Material(id=5, D_0=None, E_D=None)
    mat_2 = Material(id=2, D_0=None, E_D=None)
    mat_3 = Material(id=-1, D_0=None, E_D=None)

    my_Mats = Materials([mat_1, mat_2, mat_3])
    id_test = 1
    with pytest.raises(ValueError,
                       match="Couldn't find ID {}".format(id_test)):
        my_Mats.find_material_from_id(id_test)


def test_unused_thermal_cond():
    """
    Checks warnings when some keys are unused
    """

    mat_1 = Material(id=1, D_0=1, E_D=2, thermal_cond=2)
    my_mats = Materials([mat_1])
    with pytest.warns(
            UserWarning, match=r"thermal_cond key will be ignored"):
        my_mats.check_for_unused_properties(temp_type="expression", derived_quantities={})

    # this shouldn't throw warnings
    derived_quantities = {
            "surface_flux": [
                {
                    "field": "T",
                    "surfaces": [0, 1]
                }
                ]
    }
    # record the warnings
    with pytest.warns(None) as record:
        my_mats.check_for_unused_properties(
            temp_type="expression",
            derived_quantities=derived_quantities)

    # check that no warning were raised
    assert len(record) == 0


def test_different_ids_in_materials():
    """
    Checks that an error is raised when two materials have the same id
    """

    mat_1 = Material(id=1, D_0=1, E_D=2)
    mat_2 = Material(id=1, D_0=2, E_D=3)
    my_mats = Materials([mat_1, mat_2])
    with pytest.raises(ValueError, match=r"Some materials have the same id"):
        my_mats.check_unique_ids()


def test_unused_keys():
    """
    Checks warnings when some keys are unused
    """

    mat_1 = Material(id=1, D_0=1, E_D=2, rho=2)
    my_mats = Materials([mat_1])

    with pytest.warns(
            UserWarning, match=r"rho key will be ignored"):
        my_mats.check_for_unused_properties(
            temp_type="expression", derived_quantities={})

    mat_1.rho = None
    mat_1.heat_capacity = 2

    with pytest.warns(
            UserWarning, match=r"heat_capacity key will be ignored"):
        my_mats.check_for_unused_properties(
            temp_type="expression", derived_quantities={})


def test_non_matching_properties():
    mat_1 = Material(id=1, D_0=1, E_D=2, rho=2)
    mat_2 = Material(id=1, D_0=1, E_D=2)
    my_mats = Materials([mat_1, mat_2])
    with pytest.raises(ValueError, match=r"rho is not defined for all materials"):
        my_mats.check_consistency()


def test_check_borders():
    materials = [
        Material(id=1, D_0=None, E_D=None, borders=[0.5, 0.7]),
        Material(id=2, D_0=None, E_D=None, borders=[0, 0.5]),
            ]
    size = 0.7
    assert Materials(materials).check_borders(size) is True

    with pytest.raises(ValueError, match=r'zero'):
        size = 0.7
        materials = [
            Material(id=1, D_0=None, E_D=None, borders=[0.5, 0.7]),
            Material(id=1, D_0=None, E_D=None, borders=[0.2, 0.5]),
            ]
        Materials(materials).check_borders(size)

    with pytest.raises(ValueError, match=r'each other'):
        materials = [
            Material(id=1, D_0=None, E_D=None, borders=[0.5, 1]),
            Material(id=1, D_0=None, E_D=None, borders=[0, 0.6]),
            Material(id=1, D_0=None, E_D=None, borders=[0.6, 1]),
            ]
        size = 1
        Materials(materials).check_borders(size)

    with pytest.raises(ValueError, match=r'size'):
        materials = [
            Material(id=1, D_0=None, E_D=None, borders=[0, 1]),
        ]
        size = 3
        Materials(materials).check_borders(size)
