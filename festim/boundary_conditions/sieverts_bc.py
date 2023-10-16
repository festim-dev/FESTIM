import festim as F
import ufl
from dolfinx.fem import Expression, Function
import numpy as np


def siverts_law(T, S_0, E_S, pressure):
    """Applies the Sieverts law to compute the concentration at the boundary"""
    S = S_0 * ufl.exp(-E_S / F.k_B / T)
    return S * pressure**0.5


class SievertsBC(F.DirichletBC):
    """
    Sieverts boundary condition class

    Args:
        subdomain (festim.Subdomain): the subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition
        species (str): the name of the species

    Attributes:
        subdomain (festim.Subdomain): the subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition
        species (festim.Species or str): the name of the species
    """

    def __init__(self, subdomain, S_0, E_S, pressure, species) -> None:
        super().__init__(value=None, species=species, subdomain=subdomain)
        self.S_0 = S_0
        self.E_S = E_S
        self.pressure = pressure

    def create_value(self, mesh, function_space, temperature):
        # case 1 pressure isn't space dependent or only time dependent:
        pressure = F.as_fenics_constant(mesh=mesh, value=self.pressure)
        # case 2 pressure is space dependent

        val = Function(function_space)
        val.interpolate(
            lambda x: np.full(
                x.shape[1],
                siverts_law(
                    T=temperature,
                    S_0=F.as_fenics_constant(mesh=mesh, value=self.S_0),
                    E_S=F.as_fenics_constant(mesh=mesh, value=self.E_S),
                    pressure=pressure,
                ),
            )
        )
        print(type(val))

        self.value_fenics = val
        self.time_dependent_expressions.append(pressure)
