from festim import ThermalFlux


def test_field_is_T():
    my_flux = ThermalFlux(2)
    assert my_flux.field == "T"
