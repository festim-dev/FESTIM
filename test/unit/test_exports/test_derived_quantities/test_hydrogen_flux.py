from festim import HydrogenFlux


def test_field_is_solute():
    """
    Tests that the festim.SurfaceQuantity field is set to solute
    when festim.HydrogenFlux is used
    """

    my_flux = HydrogenFlux(2)
    assert my_flux.field == "solute"
