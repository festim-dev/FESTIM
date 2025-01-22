from dolfinx import fem

import festim as F


class FluxBCBase:
    """
    Flux boundary condition class

    Ensuring the gradient of the solution u at a boundary:
    -A * grad(u) * n = f
    where A is some material property (diffusivity for particle flux and thermal conductivity for heat flux), n is the outwards normal vector of the boundary, f is a function of space and time.


    Args:
        subdomain (festim.SurfaceSubdomain): the surface subdomain where the boundary
            condition is applied
        value (float, fem.Constant, callable): the value of the boundary condition

    Attributes:
        subdomain (festim.SurfaceSubdomain): the surface subdomain where the boundary
            condition is applied
        value (float, fem.Constant, callable): the value of the boundary condition
        value_fenics (fem.Function or fem.Constant): the value of the boundary condition in
            fenics format
        bc_expr (fem.Expression): the expression of the boundary condition that is used to
            update the value_fenics

    """

    def __init__(self, subdomain, value):
        self.subdomain = subdomain
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value is None:
            self._value = value
        elif isinstance(value, (float, int, fem.Constant, fem.Function)):
            self._value = F.Value(value)
        elif callable(value):
            self._value = F.Value(value)
        else:
            raise TypeError(
                "Value must be a float, int, fem.Constant, fem.Function, or callable"
            )


class ParticleFluxBC(FluxBCBase):
    """
    Particle flux boundary condition class
    Ensuring the gradient of the solution c at a boundary:
    -D * grad(c) * n = f
    where D is the material diffusivity, n is the outwards normal vector of the boundary, f is a function of space and time.

    Args:
        subdomain (festim.SurfaceSubdomain): the surface subdomain where the particle flux
            is applied
        value (float, fem.Constant, callable): the value of the particle flux
        species (festim.Species): the species to which the flux is applied
        species_dependent_value (dict): a dictionary containing the species that the value. Example: {"name": species}
            where "name" is the variable name in the callable value and species is a festim.Species object.

    Attributes:
        subdomain (festim.SurfaceSubdomain): the surface subdomain where the particle flux
            is applied
        value (float or fem.Constant): the value of the particle flux
        species (festim.Species): the species to which the flux is applied
        value_fenics (fem.Function or fem.Constant): the value of the particle flux in
            fenics format
        bc_expr (fem.Expression): the expression of the particle flux that is used to
            update the value_fenics
        species_dependent_value (dict): a dictionary containing the species that the value. Example: {"name": species}
            where "name" is the variable name in the callable value and species is a festim.Species object.


    Usage:
    .. testcode::

        from festim import ParticleFluxBC

        ParticleFluxBC(subdomain=my_subdomain, value=1, species="H")
        ParticleFluxBC(subdomain=my_subdomain, value=lambda x: 1 + x[0], species="H")
        ParticleFluxBC(subdomain=my_subdomain, value=lambda t: 1 + t, species="H")
        ParticleFluxBC(subdomain=my_subdomain, value=lambda T: 1 + T, species="H")
        ParticleFluxBC(subdomain=my_subdomain, value=lambda x, t: 1 + x[0] + t, species="H")
        ParticleFluxBC(subdomain=my_subdomain, value=lambda c1: 2 * c1**2, species="H", species_dependent_value={"c1": species1})
    """

    def __init__(self, subdomain, value, species, species_dependent_value={}):
        super().__init__(subdomain=subdomain, value=value)
        self.species = species
        self.species_dependent_value = species_dependent_value


class HeatFluxBC(FluxBCBase):
    """
    Heat flux boundary condition class
    Ensuring the gradient of the solution T at a boundary:
    -lambda * grad(T) * n = f
    where lambda is the thermal conductivity , n is the outwards normal vector of the boundary, f is a function of space and time.

    Args:
        subdomain (festim.SurfaceSubdomain): the surface subdomain where the heat flux
            is applied
        value (float, callable, fem.Constant): the value of the heat flux

    Attributes:
        subdomain (festim.SurfaceSubdomain): the surface subdomain where the heat flux
            is applied
        value (float, callable, fem.Constant): the value of the heat flux
        value_fenics (fem.Function or fem.Constant): the value of the heat flux in
            fenics format
        bc_expr (fem.Expression): the expression of the heat flux that is used to
            update the value_fenics


    Usage:
    .. testcode::

        from festim import HeatFluxBC
        HeatFluxBC(subdomain=my_subdomain, value=1)
        HeatFluxBC(subdomain=my_subdomain, value=lambda x: 1 + x[0])
        HeatFluxBC(subdomain=my_subdomain, value=lambda t: 1 + t)
        HeatFluxBC(subdomain=my_subdomain, value=lambda x, t: 1 + x[0] + t)
    """

    def __init__(self, subdomain, value):
        super().__init__(subdomain=subdomain, value=value)

        if self.value.temperature_dependent:
            raise ValueError("Heat flux cannot be temperature dependent")
