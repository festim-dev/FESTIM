from FESTIM import help_key, kJmol_to_eV, k_B, R


def test_help_key():
    """Tests the function help_key() with several entries
    """
    help_key("mesh_parameters")
    help_key("temperature")
    help_key("volumes")
    help_key("surfaces")
    help_key("E_p")


def test_energy_converter():
    test_values = [2, 30, 20.5, -2, -12.2]
    for energy_value in test_values:
        energy_in_eV = kJmol_to_eV(energy_value)
        expected_value = k_B*energy_value*1e3/R

        assert energy_in_eV == expected_value
