import numpy as np
import ufl
from dolfinx import fem
import festim as F


# TODO rename this to InitialConcentration and create a new base class
class InitialCondition:
    """
    Initial condition class

    Args:
        value (float, int, fem.Constant or callable): the value of the initial condition
        species (festim.Species): the species to which the condition is applied

    Attributes:
        value (float, int, fem.Constant or callable): the value of the initial condition
        species (festim.Species): the species to which the source is applied
        expr_fenics (LambdaType or fem.Expression): the value of the initial condition in
            fenics format

    Usage:
        >>> from festim import InitialCondition
        >>> InitialCondition(value=1, species=my_species)
        >>> InitialCondition(value=lambda x: 1 + x[0], species=my_species)
        >>> InitialCondition(value=lambda T: 1 + T, species=my_species)
        >>> InitialCondition(value=lambda x, T: 1 + x[0] + T, species=my_species)
    """

    def __init__(self, value, species):
        self.value = value
        self.species = species

        self.value_fenics = F.ConvertToFenicsObject(value)


class InitialTemperature:
    def __init__(self, value) -> None:
        self.value = value

        self.value_fenics = F.ConvertToFenicsObject(value)
