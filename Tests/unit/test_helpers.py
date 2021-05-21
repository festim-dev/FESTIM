from FESTIM.helpers import help_key, find_material_from_id
import pytest


def test_help_key():
    """Tests the function help_key() with several entries
    """
    help_key("mesh_parameters")
    help_key("temperature")
    help_key("volumes")
    help_key("surfaces")
    help_key("E_p")


def test_find_material_from_id_unfound_id():
    """
    Tests the material and if not previously defined
        - raise an error
    """
    materials = [
        {"id": 5},
        {"id": 2},
        {"id": -1},
    ]
    id_test = 1
    with pytest.raises(ValueError, match="Couldn't find ID {}".format(id_test)):
        find_material_from_id(materials, id_test)
