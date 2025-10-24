import numpy as np
import ufl
from dolfinx import fem

import festim as F
from festim.subdomain.surface_subdomain import SurfaceSubdomain


class FluxBCBase:
    """
    Flux boundary condition class

    Ensuring the gradient of the solution u at a boundary:

    .. math::

        -A \\nabla u \\cdot \\mathbf{n} = f

    where :math:`A` is some material property (diffusivity for particle flux and thermal
    conductivity for heat flux), :math:`\\mathbf{n}` is the outwards normal vector of
    the boundary, :math:`f` is a function of space and time.


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

    def __init__(self, subdomain: SurfaceSubdomain, value):
        self.subdomain = subdomain
        self.value = value

        self.value_fenics = None
        self.bc_expr = None

    @property
    def value_fenics(self):
        return self._value_fenics

    @value_fenics.setter
    def value_fenics(self, value):
        if value is None:
            self._value_fenics = value
            return
        if not isinstance(
            value, fem.Function | fem.Constant | np.ndarray | ufl.core.expr.Expr
        ):
            raise TypeError(
                f"Value must be a dolfinx.fem.Function, dolfinx.fem.Constant, np.ndarray or ufl.core.expr.Expr not {type(value)}"  # noqa: E501
            )
        self._value_fenics = value

    @property
    def time_dependent(self):
        if self.value is None:
            raise TypeError("Value must be given to determine if its time dependent")
        if isinstance(self.value, fem.Constant):
            return False
        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            return "t" in arguments
        else:
            return False

    @property
    def temperature_dependent(self):
        if self.value is None:
            return False
        if isinstance(self.value, fem.Constant):
            return False
        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            return "T" in arguments
        else:
            return False

    def create_value_fenics(self, mesh, temperature, t: fem.Constant):
        """Creates the value of the boundary condition as a fenics object and sets it to
        self.value_fenics.
        If the value is a constant, it is converted to a fenics.Constant.
        If the value is a function of t, it is converted to a fenics.Constant.
        Otherwise, it is converted to a ufl Expression

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            temperature (float): the temperature
            t (dolfinx.fem.Constant): the time
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, (int, float)):
            self.value_fenics = F.as_fenics_constant(mesh=mesh, value=self.value)

        elif callable(self.value):
            arguments = self.value.__code__.co_varnames

            if "t" in arguments and "x" not in arguments and "T" not in arguments:
                # only t is an argument
                if not isinstance(self.value(t=float(t)), (float, int)):
                    raise ValueError(
                        f"self.value should return a float or an int, not {type(self.value(t=float(t)))} "
                    )
                self.value_fenics = F.as_fenics_constant(
                    mesh=mesh, value=self.value(t=float(t))
                )
            else:
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = t
                if "x" in arguments:
                    kwargs["x"] = x
                if "T" in arguments:
                    kwargs["T"] = temperature

                self.value_fenics = self.value(**kwargs)

    def update(self, t):
        """Updates the flux bc value

        Args:
            t (float): the time
        """
        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            if isinstance(self.value_fenics, fem.Constant) and "t" in arguments:
                self.value_fenics.value = self.value(t=t)


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


    Examples:

        .. testsetup:: ParticleFluxBC

            from festim import SurfaceSubdomain, ParticleFluxBC, Species
            my_subdomain = SurfaceSubdomain(id=1)
            species1 = Species(name="1")

        .. testcode:: ParticleFluxBC

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

    def create_value_fenics(self, mesh, temperature, t: fem.Constant):
        """Creates the value of the boundary condition as a fenics object and sets it to
        self.value_fenics.
        If the value is a constant, it is converted to a fenics.Constant.
        If the value is a function of t, it is converted to a fenics.Constant.
        Otherwise, it is converted to a ufl Expression

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            temperature (float): the temperature
            t (dolfinx.fem.Constant): the time
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, (int, float)):
            self.value_fenics = F.as_fenics_constant(mesh=mesh, value=self.value)

        elif callable(self.value):
            arguments = self.value.__code__.co_varnames

            if "t" in arguments and "x" not in arguments and "T" not in arguments:
                # only t is an argument
                if not isinstance(self.value(t=float(t)), (float, int)):
                    raise ValueError(
                        f"self.value should return a float or an int, not {type(self.value(t=float(t)))} "
                    )
                self.value_fenics = F.as_fenics_constant(
                    mesh=mesh, value=self.value(t=float(t))
                )
            else:
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = t
                if "x" in arguments:
                    kwargs["x"] = x
                if "T" in arguments:
                    kwargs["T"] = temperature

                for name, species in self.species_dependent_value.items():
                    kwargs[name] = species.concentration

                self.value_fenics = self.value(**kwargs)


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


    Examples:

        .. testsetup:: HeatFluxBC

                from festim import SurfaceSubdomain, HeatFluxBC, Species
                my_subdomain = SurfaceSubdomain(id=1)
                species1 = Species(name="1")

        .. testcode:: HeatFluxBC

            HeatFluxBC(subdomain=my_subdomain, value=1)
            HeatFluxBC(subdomain=my_subdomain, value=lambda x: 1 + x[0])
            HeatFluxBC(subdomain=my_subdomain, value=lambda t: 1 + t)
            HeatFluxBC(subdomain=my_subdomain, value=lambda x, t: 1 + x[0] + t)
    """

    def __init__(self, subdomain, value):
        super().__init__(subdomain=subdomain, value=value)
