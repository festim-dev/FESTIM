from FESTIM import help_key, kJmol_to_eV


def test_help_key():
    """Tests the function help_key() with several entries
    """
    help_key("mesh_parameters")
    help_key("temperature")
    help_key("volumes")
    help_key("surfaces")
    help_key("E_p")


def test_energy_converter():
    energy_in_kJ = 1
    energy_in_eV = kJmol_to_eV(energy_in_kJ)

    assert 0.010364266093811426 == energy_in_eV
