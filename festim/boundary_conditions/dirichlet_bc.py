import festim as F
import ufl
from dolfinx import fem
import numpy as np


class DirichletBCBase:
    """
    Dirichlet boundary condition class
    u = value

    Args:
        subdomain (festim.Subdomain): the surface subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition

    Attributes:
        subdomain (festim.Subdomain): the surface subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition
        value_fenics (fem.Function or fem.Constant): the value of the boundary condition in
            fenics format
        bc_expr (fem.Expression): the expression of the boundary condition that is used to
            update the value_fenics
        time_dependent (bool): True if the value of the bc is time dependent

    """

    def __init__(self, subdomain, value) -> None:
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
        if not isinstance(value, (fem.Function, fem.Constant, np.ndarray)):
            raise TypeError(
                f"Value must be a dolfinx.fem.Function, dolfinx.fem.Constant, or a np.ndarray not {type(value)}"
            )
        self._value_fenics = value

    @property
    def time_dependent(self):
        if self.value is None:
            return False
        if isinstance(self.value, fem.Constant):
            return False
        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            return "t" in arguments
        else:
            return False

    def define_surface_subdomain_dofs(self, facet_meshtags, mesh, function_space):
        """Defines the facets and the degrees of freedom of the boundary
        condition

        Args:
            facet_meshtags (ddolfinx.mesh.MeshTags): the mesh tags of the
                surface facets
            mesh (dolfinx.mesh.Mesh): the mesh
            function_space (dolfinx.fem.FunctionSpaceBase): the function space
        """
        bc_facets = facet_meshtags.find(self.subdomain.id)
        bc_dofs = fem.locate_dofs_topological(function_space, mesh.fdim, bc_facets)

        return bc_dofs

    def update(self, t):
        """Updates the boundary condition value

        Args:
            t (float): the time
        """
        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            if isinstance(self.value_fenics, fem.Constant) and "t" in arguments:
                self.value_fenics.value = self.value(t=t)
            else:
                self.value_fenics.interpolate(self.bc_expr)


class FixedConcentrationBC(DirichletBCBase):
    """
    Args:
        subdomain (festim.Subdomain): the surface subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition
        species (str): the name of the species

    Attributes:
        temperature_dependent (bool): True if the value of the bc is temperature dependent

    Usage:
        >>> from festim import FixedConcentrationBC
        >>> FixedConcentrationBC(subdomain=my_subdomain, value=1, species="H")
        >>> FixedConcentrationBC(subdomain=my_subdomain, value=lambda x: 1 + x[0], species="H")
        >>> FixedConcentrationBC(subdomain=my_subdomain, value=lambda t: 1 + t, species="H")
        >>> FixedConcentrationBC(subdomain=my_subdomain, value=lambda T: 1 + T, species="H")
        >>> FixedConcentrationBC(subdomain=my_subdomain, value=lambda x, t: 1 + x[0] + t, species="H")

    """

    def __init__(self, subdomain, value, species) -> None:
        self.species = species
        super().__init__(subdomain, value)

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

    def create_value(
        self, mesh, function_space: fem.FunctionSpace, temperature, t: fem.Constant
    ):
        """Creates the value of the boundary condition as a fenics object and sets it to
        self.value_fenics.
        If the value is a constant, it is converted to a fenics.Constant.
        If the value is a function of t, it is converted to a fenics.Constant.
        Otherwise, it is converted to a fenics.Function and the
        expression of the function is stored in self.bc_expr.

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            function_space (dolfinx.fem.FunctionSpace): the function space
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
                self.value_fenics = fem.Function(function_space)
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = t
                if "x" in arguments:
                    kwargs["x"] = x
                if "T" in arguments:
                    kwargs["T"] = temperature

                # store the expression of the boundary condition
                # to update the value_fenics later
                self.bc_expr = fem.Expression(
                    self.value(**kwargs),
                    function_space.element.interpolation_points(),
                )
                self.value_fenics.interpolate(self.bc_expr)


# alias for FixedConcentrationBC
class DirichletBC(FixedConcentrationBC):
    def __init__(self, subdomain, value, species) -> None:
        super().__init__(subdomain, value, species)


class FixedTemperatureBC(DirichletBCBase):
    def __init__(self, subdomain, value):
        super().__init__(subdomain, value)

    def create_value(self, mesh, function_space: fem.FunctionSpace, t: fem.Constant):
        """Creates the value of the boundary condition as a fenics object and sets it to
        self.value_fenics.
        If the value is a constant, it is converted to a fenics.Constant.
        If the value is a function of t, it is converted to a fenics.Constant.
        Otherwise, it is converted to a fenics.Function and the
        expression of the function is stored in self.bc_expr.

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            function_space (dolfinx.fem.FunctionSpace): the function space
            t (dolfinx.fem.Constant): the time
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, (int, float)):
            self.value_fenics = F.as_fenics_constant(mesh=mesh, value=self.value)

        elif callable(self.value):
            arguments = self.value.__code__.co_varnames

            if "t" in arguments and "x" not in arguments:
                # only t is an argument
                if not isinstance(self.value(t=float(t)), (float, int)):
                    raise ValueError(
                        f"self.value should return a float or an int, not {type(self.value(t=float(t)))} "
                    )
                self.value_fenics = F.as_fenics_constant(
                    mesh=mesh, value=self.value(t=float(t))
                )
            else:
                self.value_fenics = fem.Function(function_space)
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = t
                if "x" in arguments:
                    kwargs["x"] = x

                # store the expression of the boundary condition
                # to update the value_fenics later
                self.bc_expr = fem.Expression(
                    self.value(**kwargs),
                    function_space.element.interpolation_points(),
                )
                self.value_fenics.interpolate(self.bc_expr)
