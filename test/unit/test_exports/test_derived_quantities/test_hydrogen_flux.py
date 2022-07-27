from festim import HydrogenFlux


def test_field_is_solute():
    my_flux = HydrogenFlux(2)
    assert my_flux.field == "solute"
