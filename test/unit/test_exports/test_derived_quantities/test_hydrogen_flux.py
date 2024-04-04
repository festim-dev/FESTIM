from festim import HydrogenFlux
from .tools import c_1D, c_2D, c_3D
import pytest


def test_field_is_solute():
    """
    Tests that the festim.SurfaceQuantity field is set to solute
    when festim.HydrogenFlux is used
    """

    my_flux = HydrogenFlux(2)
    assert my_flux.field == "solute"


@pytest.mark.parametrize(
    "function, expected_title",
    [
        (c_1D, "Flux surface 3: solute (H m-2 s-1)"),
        (c_2D, "Flux surface 3: solute (H m-1 s-1)"),
        (c_3D, "Flux surface 3: solute (H s-1)"),
    ],
)
def test_title_with_units(function, expected_title):
    my_flux = HydrogenFlux(3)
    my_flux.function = function
    my_flux.show_units = True

    assert my_flux.title == expected_title


def test_title_without_units():
    my_flux = HydrogenFlux(4)
    assert my_flux.title == "Flux surface 4: solute"
