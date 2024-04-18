from festim import ThermalFlux
import pytest
from .tools import c_1D, c_2D, c_3D


def test_field_is_T():
    """
    Tests that the festim.SurfaceQuantity field is set to T
    when festim.ThermalFlux is used
    """
    my_flux = ThermalFlux(2)
    assert my_flux.field == "T"


@pytest.mark.parametrize(
    "function, expected_title",
    [
        (c_1D, "Heat flux surface 5 (W m-2)"),
        (c_2D, "Heat flux surface 5 (W m-1)"),
        (c_3D, "Heat flux surface 5 (W)"),
    ],
)
def test_title_with_units(function, expected_title):
    my_flux = ThermalFlux(5)
    my_flux.function = function
    my_flux.show_units = True

    assert my_flux.title == expected_title


def test_title_without_units():
    my_flux = ThermalFlux(5)

    assert my_flux.title == "Flux surface 5: T"
