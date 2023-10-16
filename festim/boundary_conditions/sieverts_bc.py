import festim as F
import ufl
from dolfinx.fem import Expression, Function, Constant


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

    def make_fenics_obj_for_pressure(self, mesh, function_space):
        if isinstance(self.pressure, (int, float)):
            # case 1 pressure isn't space dependent or only time dependent:
            pressure = F.as_fenics_constant(mesh=mesh, value=self.pressure)
        # case 2 pressure is space dependent
        elif callable(self.pressure):
            arguments = self.pressure.__code__.co_varnames
            if "t" in arguments and "x" in arguments:
                pressure_expr = F.SpaceTimeDependentExpression(
                    function=self.pressure, t=0
                )
                pressure = Function(function_space)
                pressure.interpolate(pressure_expr.__call__)

                self.time_dependent_expressions.append(pressure_expr)

            elif "x" in arguments:
                pressure = Function(function_space)
                pressure.interpolate(self.pressure)

            elif "t" in arguments:
                pressure = F.as_fenics_constant(mesh=mesh, value=self.pressure(t=0))
                self.time_dependent_expressions.append(pressure)
        return pressure

    def create_value(self, mesh, function_space, temperature):
        pressure_as_fenics = self.make_fenics_obj_for_pressure(mesh, function_space)

        self.value_fenics = Function(function_space)
        self.bc_expr = Expression(
            siverts_law(
                T=temperature,
                S_0=self.S_0,
                E_S=self.E_S,
                pressure=pressure_as_fenics,
            ),
            function_space.element.interpolation_points(),
        )
        self.value_fenics.interpolate(self.bc_expr)

    def update(self, t):
        if callable(self.pressure):
            if "t" in self.pressure.__code__.co_varnames:
                pressure = self.time_dependent_expressions[0]
                if hasattr(pressure, "t"):
                    pressure.t = t
                elif isinstance(pressure, Constant):
                    pressure.assign(self.pressure(t=t))

                self.value_fenics.interpolate(self.bc_expr)
