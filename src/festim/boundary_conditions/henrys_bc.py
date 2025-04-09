import ufl

from festim import k_B
from festim.boundary_conditions import FixedConcentrationBC


def henrys_law(T, H_0, E_H, pressure):
    """Applies the Henry's law to compute the concentration at the boundary"""
    H = H_0 * ufl.exp(-E_H / k_B / T)
    return H * pressure


class HenrysBC(FixedConcentrationBC):
    """
    Henrys boundary condition class

    c = H * pressure
    H = H_0 * exp(-E_H / k_B / T)

    Args:
        subdomain (festim.Subdomain): the subdomain where the boundary
            condition is applied
        species (str): the name of the species
        H_0 (float or fem.Constant): the Henrys constant pre-exponential factor (H/m3/Pa)
        E_H (float or fem.Constant): the Henrys constant activation energy (eV)
        pressure (float or callable): the pressure at the boundary (Pa)

    Attributes:
        subdomain (festim.Subdomain): the subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition
        species (festim.Species or str): the name of the species
        H_0 (float or fem.Constant): the Henrys constant pre-exponential factor (H/m3/Pa)
        E_H (float or fem.Constant): the Henrys constant activation energy (eV)
        pressure (float or callable): the pressure at the boundary (Pa)

    Examples:

        .. testsetup:: HenrysBC

            from festim import HenrysBC, SurfaceSubdomain
            my_subdomain = SurfaceSubdomain(id=1)

        .. testcode:: HenrysBC

            HenrysBC(subdomain=my_subdomain, H_0=1e-6, E_H=0.2, pressure=1e5, species="H")
            HenrysBC(subdomain=my_subdomain, H_0=1e-6, E_H=0.2, pressure=lambda x: 1e5 + x[0], species="H")
            HenrysBC(subdomain=my_subdomain, H_0=1e-6, E_H=0.2, pressure=lambda t: 1e5 + t, species="H")
            HenrysBC(subdomain=my_subdomain, H_0=1e-6, E_H=0.2, pressure=lambda T: 1e5 + T, species="H")
            HenrysBC(subdomain=my_subdomain, H_0=1e-6, E_H=0.2, pressure=lambda x, t: 1e5 + x[0] + t, species="H")
    """

    def __init__(self, subdomain, H_0, E_H, pressure, species) -> None:
        # TODO find a way to have H_0 and E_H as fem.Constant
        # maybe in create_value()
        self.H_0 = H_0
        self.E_H = E_H
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
                ("x",): lambda T, x=None: henrys_law(
                    T, self.H_0, self.E_H, self.pressure(x=x)
                ),
                ("t",): lambda T, t=None: henrys_law(
                    T, self.H_0, self.E_H, self.pressure(t=t)
                ),
                ("T",): lambda T: henrys_law(T, self.H_0, self.E_H, self.pressure(T=T)),
                ("t", "x"): lambda T, x=None, t=None: henrys_law(
                    T, self.H_0, self.E_H, self.pressure(x=x, t=t)
                ),
                ("T", "x"): lambda T, x=None: henrys_law(
                    T, self.H_0, self.E_H, self.pressure(x=x, T=T)
                ),
                ("T", "t"): lambda T, t=None: henrys_law(
                    T, self.H_0, self.E_H, self.pressure(t=t, T=T)
                ),
                ("T", "t", "x"): lambda T, x=None, t=None: henrys_law(
                    T, self.H_0, self.E_H, self.pressure(x=x, t=t, T=T)
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
            return lambda T: henrys_law(T, self.H_0, self.E_H, self.pressure)
