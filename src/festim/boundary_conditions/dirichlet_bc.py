from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import ufl
import ufl.argument
import ufl.core
import ufl.core.expr
import ufl.indexed
from dolfinx import fem
from dolfinx import mesh as _mesh

from festim import helpers
from festim import subdomain as _subdomain
from festim.species import Species


class DirichletBCBase:
    """Dirichlet boundary condition class u = value.

    Args:
        subdomain: The surface subdomain where the boundary condition is applied
        value: The value of the boundary condition
        enforce_weakly: Whether to enforce the boundary condition weakly using Nitsche's
            method. Defaults to False.
        penalty: The dimensionless penalty parameter to use if ``enforce_weakly`` is
            True. The Nitsche penalty term scales as ``penalty * D / h``, so a value of
            order 10-100 is appropriate regardless of the material or the mesh size.
            Defaults to None

    Attributes:
        subdomain: The surface subdomain where the boundary condition is applied
        value: The value of the boundary condition
        value_fenics: The value of the boundary condition in fenics format
        bc_expr: The expression of the boundary condition that is used to
            update the `value_fenics`
        enforce_weakly: Whether to enforce the boundary condition weakly using Nitsche's
            method.
        penalty: The penalty parameter to use if ``enforce_weakly`` is True.
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
        enforce_weakly: bool = False,
        penalty: float | None = None,
    ):
        self.subdomain = subdomain
        self.value = value
        self.enforce_weakly = enforce_weakly
        self.penalty = penalty

        self.value_fenics = None
        self.bc_expr = None

    @property
    def value_fenics(self):
        return self._value_fenics

    @value_fenics.setter
    def value_fenics(
        self,
        value: None | fem.Function | fem.Constant | np.ndarray | ufl.core.expr.Expr,
    ):
        if value is None:
            self._value_fenics = value
            return
        # a ufl expression is allowed for weakly enforced BCs only: it cannot be
        # interpolated into a fem.Function, which is what a strong BC needs. This is
        # how a coupling to an enclosure pressure (a real-space unknown) gets in.
        if not isinstance(
            value, (fem.Function, fem.Constant, np.ndarray, ufl.core.expr.Expr)
        ):
            # FIXME: Should we allow sending in a callable here?
            raise TypeError(
                "Value must be a dolfinx.fem.Function, dolfinx.fem.Constant, a np.ndarray or a ufl expression, not"  # noqa: E501
                + f"{type(value)}"
            )
        self._value_fenics = value

    @property
    def time_dependent(self) -> bool:
        """Returns true if the value of the boundary condition is time dependent."""
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

        Given the input meshtags, find all facets matching the boundary
        condition subdomain ID,
        and locate all DOFs associated with the input function space(s).

        Note:
            For sub-spaces, a tuple of sub-spaces are expected as input, and a
            tuple of arrays
            associated to each of the function spaces are returned.

        Args:
            facet_meshtags: MeshTags describing some facets in the domain
            mesh:
            function_space: The function space or a tuple of function spaces:
            (sub, collapsed)
        """
        mesh = (
            function_space[0].mesh
            if isinstance(function_space, tuple)
            else function_space.mesh
        )
        if facet_meshtags.topology != mesh.topology._cpp_object:
            raise ValueError(
                "Mesh of function-space is not the same as the one used for the meshtags"  # noqa: E501
            )
        if mesh.topology.dim - 1 != facet_meshtags.dim:
            raise ValueError(
                f"Meshtags of dimension {facet_meshtags.dim}, expected {mesh.topology.dim - 1}"  # noqa: E501
            )
        bc_dofs = fem.locate_dofs_topological(
            function_space, facet_meshtags.dim, facet_meshtags.find(self.subdomain.id)
        )

        return bc_dofs

    def update(self, t: float):
        """Updates the boundary condition value.

        Args:
            t: the time
        """
        if isinstance(self.value_fenics, ufl.core.expr.Expr) and not isinstance(
            self.value_fenics, (fem.Function, fem.Constant)
        ):
            # a pure ufl value is built on live coefficients (the temperature, an
            # enclosure pressure), so it already reflects their current values
            return
        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            if isinstance(self.value_fenics, fem.Constant) and "t" in arguments:
                self.value_fenics.value = self.value(t=t)
            else:
                self.value_fenics.interpolate(self.bc_expr)
        elif self.bc_expr is not None:
            self.value_fenics.interpolate(self.bc_expr)

    def numerical_flux(
        self,
        u: fem.Function | ufl.indexed.Indexed,
        D: ufl.core.expr.Expr | fem.Function | fem.Constant,
        mesh,
    ) -> ufl.core.expr.Expr:
        """Returns the numerical flux leaving the domain through the surface of the BC.

        Nitsche's method only enforces ``u = value`` weakly, so ``u - value`` does not
        vanish on the boundary and the raw gradient ``-D grad(u).n`` is not the quantity
        the discrete scheme conserves. The conserved quantity is this numerical flux,
        which includes the penalty contribution. Use it (rather than the raw gradient)
        whenever the flux through a weakly enforced boundary feeds another equation, so
        that the discrete balance is exact.

        Args:
            u: the solution function associated to the species for which the BC
                is applied
            D: the diffusion coefficient of the species at this surface
            mesh: the mesh the surface belongs to

        Returns:
            the numerical flux, positive when leaving the domain
        """
        n = ufl.FacetNormal(mesh)
        h = ufl.Circumradius(mesh)  # FIXME this doesn't work for rectangles
        alpha = self.penalty
        assert alpha is not None, (
            "Penalty parameter must be given for weakly enforced Dirichlet BCs"
        )
        assert self.value_fenics is not None, (
            "value_fenics must be defined for weakly enforced Dirichlet BCs"
        )

        # the penalty scales with D/h, like the consistency term it has to dominate,
        # which makes ``penalty`` a dimensionless parameter independent of the material
        # and of the mesh size
        return -D * ufl.inner(n, ufl.grad(u)) + alpha * D / h * (u - self.value_fenics)

    def weak_formulation(
        self,
        u: fem.Function | ufl.indexed.Indexed,
        v: ufl.argument.Argument | ufl.indexed.Indexed,
        ds: ufl.Measure,
        D: ufl.core.expr.Expr | fem.Function | fem.Constant,
    ) -> ufl.core.expr.Expr:
        """
        Returns the symmetric Nitsche weak formulation for the BC
        This follows the dolfinx tutorial
        https://jsdokken.com/dolfinx-tutorial/chapter1/nitsche.html

        Args:
            u: the solution function associated to the species for which the BC
                is applied
            v: the test function
            ds: the surface measure
            D: the diffusion coefficient of the species at this surface. It multiplies
                the consistency and symmetry terms, without which the scheme is only
                consistent when ``D == 1``.

        Returns:
            the weak formulation
        """
        mesh = ds.ufl_domain()
        n = ufl.FacetNormal(mesh)
        ds_bc = ds(self.subdomain.id)

        # consistency + penalty, grouped as the numerical flux so that the flux the
        # scheme conserves is defined in a single place
        form = self.numerical_flux(u, D, mesh) * v * ds_bc

        # symmetry term, making the bilinear form symmetric (and the L2 error optimal)
        form += -D * ufl.inner(n, ufl.grad(v)) * (u - self.value_fenics) * ds_bc

        return form


class FixedConcentrationBC(DirichletBCBase):
    """
    Args:
        subdomain (festim.Subdomain): the surface subdomain where the boundary
            condition is applied
        value: The value of the boundary condition. It can be a function of
        space and/or time
        species: The name of the species
        enforce_weakly: Whether to enforce the boundary condition weakly using Nitsche's
            method. Defaults to False.
        penalty: The dimensionless penalty parameter to use if ``enforce_weakly`` is
            True. The Nitsche penalty term scales as ``penalty * D / h``, so a value of
            order 10-100 is appropriate regardless of the material or the mesh size.
            Defaults to None

    Attributes:
        temperature_dependent (bool): True if the value of the bc is
        temperature dependent

    Examples:

        .. testsetup:: FixedConcentrationBC

            from festim import FixedConcentrationBC, SurfaceSubdomain
            my_subdomain = SurfaceSubdomain(id=1)

        .. testcode:: FixedConcentrationBC

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

    #: number of particles of ``species`` in the solid per particle of the gas species
    #: the BC is coupled to. 1 by default (Henry's law, where the gas molecule
    #: dissolves as such); 2 for Sieverts' law, where a diatomic molecule dissolves as
    #: two atoms. Only used when the BC is coupled to a
    #: :py:class:`festim.GasSpecies`.
    stoichiometry: float = 1

    #: the gas species this BC is coupled to, set by the problem when the ``pressure``
    #: of the BC is a :py:class:`festim.GasSpecies`
    _gas_species = None

    def __init__(
        self,
        subdomain: _subdomain.SurfaceSubdomain,
        value: np.ndarray | fem.Constant | int | float | Callable,
        species: Species,
        enforce_weakly: bool = False,
        penalty: float | None = None,
    ):
        self.species = species
        super().__init__(
            subdomain, value, enforce_weakly=enforce_weakly, penalty=penalty
        )

    def create_value_ufl(self, temperature: float | fem.Constant | fem.Function):
        """Creates the value of the boundary condition as a pure ufl expression and
        sets it to ``self.value_fenics``.

        Unlike :py:meth:`create_value`, this never interpolates the value into a
        ``dolfinx.fem.Function``. It is required when the value depends on an unknown of
        the problem, such as the pressure of a :py:class:`festim.Enclosure`, which lives
        in a real function space and cannot be interpolated. Such a BC can therefore
        only be enforced weakly.

        Args:
            temperature: the temperature

        Raises:
            ValueError: if the value is not a callable of the temperature only
        """
        if not callable(self.value):
            raise ValueError(
                f"The value of {self} is not callable, so it does not need "
                "create_value_ufl. Use create_value instead."
            )

        arguments = self.value.__code__.co_varnames
        if "x" in arguments or "t" in arguments:
            raise ValueError(
                "A boundary condition coupled to an enclosure pressure cannot also "
                "depend on space or time, because its value cannot be interpolated. "
                f"The value of {self} depends on {sorted(set(arguments) & {'x', 't'})}."
            )

        self.value_fenics = self.value(T=temperature)

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
        self.value_fenics. If the value is a constant, it is converted to a
        `dolfinx.fem.Constant`. If the value is a function of t, it is converted to
        `dolfinx.fem.Constant`. Otherwise, it is converted to a
        `dolfinx.fem.Function`.Function and the expression of the function is stored in
        `bc_expr`.

        Args:
            function_space: the function space
            temperature: The temperature
            t: the time
            K_S: The solubility of the species. If provided, the value of the
            boundary condition
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
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = t
                if "x" in arguments:
                    kwargs["x"] = x
                if "T" in arguments:
                    kwargs["T"] = temperature

                self.value_fenics = fem.Function(function_space)

                # store the expression of the boundary condition
                # to update the value_fenics later
                assert isinstance(self.value(**kwargs), ufl.core.expr.Expr), (
                    f"{type(self.value(**kwargs))}"
                )
                self.bc_expr = fem.Expression(
                    self.value(**kwargs),
                    function_space.element.interpolation_points,
                )

                self.value_fenics.interpolate(self.bc_expr)

        # if K_S is provided, divide the value by K_S (change of variable method)
        if K_S is not None:
            if isinstance(self.value, (int, float)):
                val_as_cst = helpers.as_fenics_constant(mesh=mesh, value=self.value)
                self.bc_expr = fem.Expression(
                    val_as_cst / K_S,
                    function_space.element.interpolation_points,
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
                        function_space.element.interpolation_points,
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
                        function_space.element.interpolation_points,
                    )
                    self.value_fenics.interpolate(self.bc_expr)


# alias for FixedConcentrationBC
DirichletBC = FixedConcentrationBC


class FixedTemperatureBC(DirichletBCBase):
    def create_value(self, function_space: fem.FunctionSpace, t: fem.Constant):
        """Creates the value of the boundary condition as a fenics object and sets it to
        self.value_fenics. If the value is a constant, it is converted to a
        `dolfinx.fem.Constant`. If the value is a function of t, it is converted to a
        `dolfinx.fem.Constant`. Otherwise, it is converted to a` dolfinx.fem.Function`
        and the expression of the function is stored in `bc_expr`.

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
                    function_space.element.interpolation_points,
                )
                self.value_fenics.interpolate(self.bc_expr)
