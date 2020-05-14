from FESTIM.helpers import help_key


def test_help_key():
    """Tests the function help_key() with several entries
    """
    help_key("mesh_parameters")
    help_key("temperature")
    help_key("volumes")
    help_key("surfaces")
    help_key("E_p")
