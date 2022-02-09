from FESTIM import x
import sympy as sp
import numpy as np


class Source:
    """Volumetric source term.

    Attributes:
        value (sympy.Add, float): the value of the volumetric source term
        volume (int): the volume in which the source is applied
        field (str): the field on which the source is applied ("0", "solute",
            "1", "T")
    """
    def __init__(self, value, volume, field) -> None:
        """Inits Source

        Args:
            value (sympy.Add, float): the value of the volumetric source term
            volume (int): the volume in which the source is applied
            field (str): the field on which the source is applied ("0", "solute",
                "1", "T")
        """
        self.value = value
        self.volume = volume
        self.field = field


class ImplantationFlux(Source):
    """
    Implantation flux [add description here!]

    Usage:
    my_source = ImplantationFlux(
        flux=2*FESTIM.x * (FESTIM.t < 10),
        imp_depth=5e-9, width=5e-9, volume=1)


    Attributes:
        flux (float): The flux of the implatation source (m2/s)
        imp_depth (float): Depth of implantation (m)
        width (float): The dispersion of the ion beam (m)
    """
    def __init__(self, flux, imp_depth, width, volume, field="0"):
        """
        Args:
            flux (float): The flux of the implatation source (m2/s)
            imp_depth (float): Depth of implantation (m)
            width (float): The dispersion of the ion beam (m)
            volume (int): the volume in which the source is applied
        """
        self.volume = volume
        self.field = field
        self.flux = flux
        self.imp_depth = imp_depth
        self.width = width
        distribution = 1/(self.width*(2*np.pi)**0.5) * \
            sp.exp(-0.5*((x-self.imp_depth)/self.width)**2)
        value = self.flux*distribution
        super().__init__(value, volume, field)
