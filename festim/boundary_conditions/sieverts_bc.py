import festim as F
import ufl
from dolfinx.fem import Expression, Function, Constant


def sieverts_law(T, S_0, E_S, pressure):
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
        self.S_0 = S_0
        self.E_S = E_S
        self.pressure = pressure

        # construct value callable based on args of pressure
        args_value_fun = ["T"]
        if callable(self.pressure):
            if "t" in self.pressure.__code__.co_varnames:
                args_value_fun.append("t")
            if "x" in self.pressure.__code__.co_varnames:
                args_value_fun.append("x")

        # FIXME
        def value_fun(*args):
            return sieverts_law(
                T=args[0], S_0=self.S_0, E_S=self.E_S, pressure=pressure(*args)
            )

        super().__init__(value=value_fun, species=species, subdomain=subdomain)
