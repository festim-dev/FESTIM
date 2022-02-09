from FESTIM import Source, x
import sympy as sp
import numpy as np


class ImplantationFlux(Source):
    """
    Implantation flux represented by a volumetric mobile particle source
    representing implantaion of ions with a 1D gaussian distribution.

    Only can be used in 1D cases

    Usage:
    my_source = ImplantationFlux(
        flux=2*FESTIM.x * (FESTIM.t < 10),
        imp_depth=5e-9, width=5e-9, volume=1)


    Attributes:
        flux (float, sympy.expr): The flux of the implatation source (m-2 s-1)
        imp_depth (float, sympy.expr): Depth of implantation (m)
        width (float, sympy.expr): The dispersion of the ion beam (m)
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
