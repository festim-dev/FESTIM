import FESTIM
import sympy as sp
import numpy as np
import pytest


def test_implantation_flux_attributes():
    """
    Checks all attributes of the implantation flux class
    """
    flux_val = 1
    imp_depth_val = 5e-9
    width_val = 5e-9
    distribution = 1/(width_val*(2*np.pi)**0.5) * \
        sp.exp(-0.5*((FESTIM.x-imp_depth_val)/width_val)**2)
    expected_value = flux_val*distribution

    my_source = FESTIM.ImplantationFlux(flux=flux_val, imp_depth=imp_depth_val,
                                        width=width_val, volume=1)
    flux = my_source.flux
    imp_depth = my_source.imp_depth
    width = my_source.width
    value = my_source.value

    assert flux == flux_val
    assert imp_depth == imp_depth_val
    assert width == width_val
    assert expected_value == value


def test_implantation_flux_with_time_dependancy():
    """
    Test to check flux can be defined with a time dependance
    """
    flux = 1*(FESTIM.t < 10)
    imp_depth = 5e-9
    width = 5e-9
    distribution = 1/(width*(2*np.pi)**0.5) * \
        sp.exp(-0.5*((FESTIM.x-imp_depth)/width)**2)
    expected_value = flux*distribution

    my_source = FESTIM.ImplantationFlux(flux=1*(FESTIM.t < 10),
                                        imp_depth=5e-9, width=5e-9, volume=1)
    value = my_source.value
    assert value == expected_value
