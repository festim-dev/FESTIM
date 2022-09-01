from festim import Source, x
import sympy as sp
import numpy as np


class ImplantationFlux(Source):
    """
    Implantation flux represented by a volumetric mobile particle source
    with a 1D gaussian distribution in the x direction.

    Args:
        flux (float, sympy.Expr): The flux of the implatation source (m-2 s-1)
        imp_depth (float, sympy.Expr): The implantation depth (m)
        width (float, sympy.Expr): The standard deviation of the ion beam (m)
        volume (int): the volume in which the source is applied

    Attributes:
        flux (float, sympy.Expr): The flux of the implatation source (m-2 s-1)
        imp_depth (float, sympy.Expr): The implantation depth (m)
        width (float, sympy.Expr): The standard deviation of the ion beam (m)

    Example::

        my_source = ImplantationFlux(
            flux=2*festim.x * (festim.t < 10),
            imp_depth=5e-9,
            width=5e-9,
            volume=1,
        )
    """

    def __init__(self, flux, imp_depth, width, volume):
        self.flux = flux
        self.imp_depth = imp_depth
        self.width = width
        distribution = (
            1
            / (self.width * (2 * np.pi) ** 0.5)
            * sp.exp(-0.5 * ((x - self.imp_depth) / self.width) ** 2)
        )
        value = self.flux * distribution
        super().__init__(value, volume, field="0")
