from festim import ThermalFlux


def test_field_is_T():
    """
    Tests that the festim.SurfaceQuantity field is set to T
    when festim.ThermalFlux is used
    """
    my_flux = ThermalFlux(2)
    assert my_flux.field == "T"
