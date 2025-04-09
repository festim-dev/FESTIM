import ufl

from festim import k_B
from festim.boundary_conditions import FixedConcentrationBC


def sieverts_law(T, S_0, E_S, pressure):
    """Applies the Sieverts law to compute the concentration at the boundary"""
    S = S_0 * ufl.exp(-E_S / k_B / T)
    return S * pressure**0.5


class SievertsBC(FixedConcentrationBC):
    """
    Sieverts boundary condition class

    c = S * sqrt(pressure)
    S = S_0 * exp(-E_S / k_B / T)

    Args:
        subdomain (festim.Subdomain): the subdomain where the boundary
            condition is applied
        species (str): the name of the species
        S_0 (float or fem.Constant): the Sieverts constant pre-exponential factor (H/m3/Pa0.5)
        E_S (float or fem.Constant): the Sieverts constant activation energy (eV)
        pressure (float or callable): the pressure at the boundary (Pa)

    Attributes:
        subdomain (festim.Subdomain): the subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition
        species (festim.Species or str): the name of the species
        S_0 (float or fem.Constant): the Sieverts constant pre-exponential factor (H/m3/Pa0.5)
        E_S (float or fem.Constant): the Sieverts constant activation energy (eV)
        pressure (float or callable): the pressure at the boundary (Pa)

    Examples:

        .. testsetup:: SievertsBC

            from festim import SievertsBC, SurfaceSubdomain
            my_subdomain = SurfaceSubdomain(id=1)

        .. testcode:: SievertsBC

            SievertsBC(subdomain=my_subdomain, S_0=1e-6, E_S=0.2, pressure=1e5, species="H")
            SievertsBC(subdomain=my_subdomain, S_0=1e-6, E_S=0.2, pressure=lambda x: 1e5 + x[0], species="H")
            SievertsBC(subdomain=my_subdomain, S_0=1e-6, E_S=0.2, pressure=lambda t: 1e5 + t, species="H")
            SievertsBC(subdomain=my_subdomain, S_0=1e-6, E_S=0.2, pressure=lambda T: 1e5 + T, species="H")
            SievertsBC(subdomain=my_subdomain, S_0=1e-6, E_S=0.2, pressure=lambda x, t: 1e5 + x[0] + t, species="H")
    """

    def __init__(self, subdomain, S_0, E_S, pressure, species) -> None:
        # TODO find a way to have S_0 and E_S as fem.Constant
        # maybe in create_value()
        self.S_0 = S_0
        self.E_S = E_S
        self.pressure = pressure

        value = self.create_new_value_function()

        super().__init__(value=value, species=species, subdomain=subdomain)

    def create_new_value_function(self):
        """Creates a new value function based on the pressure attribute

        Raises:
            ValueError: if the pressure function is not supported

        Returns:
            callable: the value function
        """
        if callable(self.pressure):
            arg_combinations = {
                ("x",): lambda T, x=None: sieverts_law(
                    T, self.S_0, self.E_S, self.pressure(x=x)
                ),
                ("t",): lambda T, t=None: sieverts_law(
                    T, self.S_0, self.E_S, self.pressure(t=t)
                ),
                ("T",): lambda T: sieverts_law(
                    T, self.S_0, self.E_S, self.pressure(T=T)
                ),
                ("t", "x"): lambda T, x=None, t=None: sieverts_law(
                    T, self.S_0, self.E_S, self.pressure(x=x, t=t)
                ),
                ("T", "x"): lambda T, x=None: sieverts_law(
                    T, self.S_0, self.E_S, self.pressure(x=x, T=T)
                ),
                ("T", "t"): lambda T, t=None: sieverts_law(
                    T, self.S_0, self.E_S, self.pressure(t=t, T=T)
                ),
                ("T", "t", "x"): lambda T, x=None, t=None: sieverts_law(
                    T, self.S_0, self.E_S, self.pressure(x=x, t=t, T=T)
                ),
            }

            # get the arguments of the pressure function
            args = self.pressure.__code__.co_varnames
            key = tuple(sorted(args))

            # get the lambda function based on the argument combination
            if key not in arg_combinations:
                raise ValueError("pressure function not supported")

            func = arg_combinations[key]

            return func
        else:
            return lambda T: sieverts_law(T, self.S_0, self.E_S, self.pressure)
