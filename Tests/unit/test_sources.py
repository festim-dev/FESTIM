import FESTIM
import sympy as sp
import numpy as np


def test_implantation_flux_attributes():
    """
    Checks all attributes of the ImplantationFlux class
    """
    flux = 1
    imp_depth = 5e-9
    width = 5e-9
    distribution = 1/(width*(2*np.pi)**0.5) * \
        sp.exp(-0.5*((FESTIM.x-imp_depth)/width)**2)
    expected_value = sp.printing.ccode(flux*distribution)

    my_source = FESTIM.ImplantationFlux(flux=flux, imp_depth=imp_depth,
                                        width=width, volume=1)

    assert my_source.flux == flux
    assert my_source.imp_depth == imp_depth
    assert my_source.width == width
    assert my_source.value._cppcode == expected_value


def test_implantation_flux_with_time_dependancy():
    """
    Checks that ImplantationFlux has the correct value attribute when using
    time dependdant arguments
    """
    flux = 1*(FESTIM.t < 10)
    imp_depth = 5e-9
    width = 5e-9
    distribution = 1/(width*(2*np.pi)**0.5) * \
        sp.exp(-0.5*((FESTIM.x-imp_depth)/width)**2)
    expected_value = sp.printing.ccode(flux*distribution)

    my_source = FESTIM.ImplantationFlux(flux=1*(FESTIM.t < 10),
                                        imp_depth=5e-9, width=5e-9, volume=1)

    assert my_source.value._cppcode == expected_value
