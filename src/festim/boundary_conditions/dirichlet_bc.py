from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import fem
from dolfinx import mesh as _mesh
import ufl.core
import ufl.core.expr

from festim import helpers
from festim import subdomain as _subdomain
from festim.species import Species


class DirichletBCBase:
    """
    Dirichlet boundary condition class
    u = value

    Args:
        subdomain: The surface subdomain where the boundary condition is applied
        value: The value of the boundary condition

    Attributes:
        subdomain: The surface subdomain where the boundary condition is applied
        value: The value of the boundary condition
        value_fenics: The value of the boundary condition in fenics format
        bc_expr: The expression of the boundary condition that is used to
            update the `value_fenics`

    """

    subdomain: _subdomain.SurfaceSubdomain
    value: (
        np.ndarray
        | fem.Constant
        | int
        | float
        | Callable[[np.ndarray], np.ndarray]
        | Callable[[np.ndarray, float], np.ndarray]
        | Callable[[float], float]
    )
    value_fenics: None | fem.Function | fem.Constant | np.ndarray | float
    bc_expr: fem.Expression

    def __init__(
        self,
        subdomain: _subdomain.SurfaceSubdomain,
        value: np.ndarray | fem.Constant | int | float | Callable,
    ):
        self.subdomain = subdomain
        self.value = value

        self.value_fenics = None
        self.bc_expr = None

    @property
    def value_fenics(self):
        return self._value_fenics

    @value_fenics.setter
    def value_fenics(self, value: None | fem.Function | fem.Constant | np.ndarray):
        if value is None:
            self._value_fenics = value
            return
        if not isinstance(value, (fem.Function, fem.Constant, np.ndarray)):
            # FIXME: Should we allow sending in a callable here?
            raise TypeError(
                "Value must be a dolfinx.fem.Function, dolfinx.fem.Constant, or a np.ndarray not"
                + f"{type(value)}"
            )
        self._value_fenics = value

    @property
    def time_dependent(self) -> bool:
        """Returns true if the value of the boundary condition is time dependent"""
        if self.value is None:
            return False
        if isinstance(self.value, fem.Constant):
            return False
        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            return "t" in arguments
        else:
            return False

    def define_surface_subdomain_dofs(
        self,
        facet_meshtags: _mesh.MeshTags,
        function_space: fem.FunctionSpace | tuple[fem.FunctionSpace, fem.FunctionSpace],
    ) -> npt.NDArray[np.int32] | tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Defines the facets and the degrees of freedom of the boundary condition.

        Given the input meshtags, find all facets matching the boundary condition subdomain ID,
        and locate all DOFs associated with the input function space(s).

        Note:
            For sub-spaces, a tuple of sub-spaces are expected as input, and a tuple of arrays
            associated to each of the function spaces are returned.

        Args:
            facet_meshtags: MeshTags describing some facets in the domain
            mesh:
            function_space: The function space or a tuple of function spaces: (sub, collapsed)
        """
        mesh = (
            function_space[0].mesh
            if isinstance(function_space, tuple)
            else function_space.mesh
        )
        if facet_meshtags.topology != mesh.topology._cpp_object:
            raise ValueError(
                "Mesh of function-space is not the same as the one used for the meshtags"
            )
        if mesh.topology.dim - 1 != facet_meshtags.dim:
            raise ValueError(
                f"Meshtags of dimension {facet_meshtags.dim}, expected {mesh.topology.dim-1}"
            )
        bc_dofs = fem.locate_dofs_topological(
            function_space, facet_meshtags.dim, facet_meshtags.find(self.subdomain.id)
        )

        return bc_dofs

    def update(self, t: float):
        """Updates the boundary condition value

        Args:
            t: the time
        """
        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            if isinstance(self.value_fenics, fem.Constant) and "t" in arguments:
                self.value_fenics.value = self.value(t=t)
            else:
                self.value_fenics.interpolate(self.bc_expr)
        elif self.bc_expr is not None:
            self.value_fenics.interpolate(self.bc_expr)


class FixedConcentrationBC(DirichletBCBase):
    """
    Args:
        subdomain (festim.Subdomain): the surface subdomain where the boundary
            condition is applied
        value: The value of the boundary condition. It can be a function of space and/or time
        species: The name of the species

    Attributes:
        temperature_dependent (bool): True if the value of the bc is temperature dependent

    Examples:

        .. highlight:: python
        .. code-block:: python

            from festim import FixedConcentrationBC
            FixedConcentrationBC(subdomain=my_subdomain, value=1, species="H")
            FixedConcentrationBC(subdomain=my_subdomain,
                                 value=lambda x: 1 + x[0], species="H")
            FixedConcentrationBC(subdomain=my_subdomain,
                                 value=lambda t: 1 + t, species="H")
            FixedConcentrationBC(subdomain=my_subdomain,
                                 value=lambda T: 1 + T, species="H")
            FixedConcentrationBC(subdomain=my_subdomain,
                                 value=lambda x, t: 1 + x[0] + t, species="H")

    """

    species: Species

    def __init__(
        self,
        subdomain: _subdomain.SurfaceSubdomain,
        value: np.ndarray | fem.Constant | int | float | Callable,
        species: Species,
    ):
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
        self,
        function_space: fem.FunctionSpace,
        temperature: float | fem.Constant,
        t: float | fem.Constant,
        K_S: fem.Function = None,
    ):
        """Creates the value of the boundary condition as a fenics object and sets it to
        self.value_fenics.
        If the value is a constant, it is converted to a `dolfinx.fem.Constant`.
        If the value is a function of t, it is converted to  `dolfinx.fem.Constant`.
        Otherwise, it is converted to a `dolfinx.fem.Function`.Function and the
        expression of the function is stored in `bc_expr`.

        Args:
            function_space: the function space
            temperature: The temperature
            t: the time
            K_S: The solubility of the species. If provided, the value of the boundary condition
                is divided by K_S (change of variable method).
        """
        mesh = function_space.mesh
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, (int, float)):
            self.value_fenics = helpers.as_fenics_constant(mesh=mesh, value=self.value)

        elif callable(self.value):
            arguments = self.value.__code__.co_varnames

            if "t" in arguments and "x" not in arguments and "T" not in arguments:
                # only t is an argument
                if not isinstance(self.value(t=float(t)), (float, int)):
                    raise ValueError(
                        "self.value should return a float or an int, not "
                        + f"{type(self.value(t=float(t)))} "
                    )
                self.value_fenics = helpers.as_fenics_constant(
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

        # if K_S is provided, divide the value by K_S (change of variable method)
        if K_S is not None:
            if isinstance(self.value, (int, float)):
                val_as_cst = helpers.as_fenics_constant(mesh=mesh, value=self.value)
                self.bc_expr = fem.Expression(
                    val_as_cst / K_S,
                    function_space.element.interpolation_points(),
                )
                self.value_fenics = fem.Function(function_space)
                self.value_fenics.interpolate(self.bc_expr)

            elif callable(self.value):
                arguments = self.value.__code__.co_varnames

                if "t" in arguments and "x" not in arguments and "T" not in arguments:
                    # only t is an argument

                    # check that value returns a ufl expression
                    if not isinstance(self.value(t=t), (ufl.core.expr.Expr)):
                        raise ValueError(
                            "self.value should return a ufl expression"
                            + f"{type(self.value(t=t))} "
                        )

                    self.bc_expr = fem.Expression(
                        self.value(t=t) / K_S,
                        function_space.element.interpolation_points(),
                    )
                    self.value_fenics = fem.Function(function_space)
                    self.value_fenics.interpolate(self.bc_expr)
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
                        self.value(**kwargs) / K_S,
                        function_space.element.interpolation_points(),
                    )
                    self.value_fenics.interpolate(self.bc_expr)


# alias for FixedConcentrationBC
DirichletBC = FixedConcentrationBC


class FixedTemperatureBC(DirichletBCBase):
    def create_value(self, function_space: fem.FunctionSpace, t: fem.Constant):
        """Creates the value of the boundary condition as a fenics object and sets it to
        self.value_fenics.
        If the value is a constant, it is converted to a `dolfinx.fem.Constant`.
        If the value is a function of t, it is converted to a `dolfinx.fem.Constant`.
        Otherwise, it is converted to a` dolfinx.fem.Function` and the
        expression of the function is stored in `bc_expr`.

        Args:
            function_space: the function space
            t: the time
        """
        mesh = function_space.mesh
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, (int, float)):
            self.value_fenics = helpers.as_fenics_constant(mesh=mesh, value=self.value)

        elif callable(self.value):
            arguments = self.value.__code__.co_varnames

            if "t" in arguments and "x" not in arguments:
                # only t is an argument
                if not isinstance(self.value(t=float(t)), (float, int)):
                    raise ValueError(
                        "self.value should return a float or an int, not "
                        + f"{type(self.value(t=float(t)))} "
                    )
                self.value_fenics = helpers.as_fenics_constant(
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
