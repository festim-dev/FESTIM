from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import fem
from dolfinx import mesh as _mesh

from festim import helpers
from festim import subdomain as _subdomain


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

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value is None:
            self._value = value
        elif isinstance(value, (float, int, fem.Constant, fem.Function)):
            self._value = helpers.Value(value)
        elif callable(value):
            self._value = helpers.Value(value)
        else:
            raise TypeError(
                "Value must be a float, int, fem.Constant, fem.Function, or callable"
            )

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

    species: str

    def __init__(
        self,
        subdomain: _subdomain.SurfaceSubdomain,
        value: np.ndarray | fem.Constant | int | float | Callable,
        species: str,
    ):
        self.species = species
        super().__init__(subdomain, value)


# alias for FixedConcentrationBC
DirichletBC = FixedConcentrationBC


class FixedTemperatureBC(DirichletBCBase):
    """
    Args:
        subdomain (festim.Subdomain): the surface subdomain where the boundary
            condition is applied
        value: The value of the boundary condition. It can be a function of space and/or time

    Examples:

        .. highlight:: python
        .. code-block:: python

            from festim import FixedTemperatureBC
            FixedTemperatureBC(subdomain=my_subdomain, value=1)
            FixedTemperatureBC(subdomain=my_subdomain,
                                 value=lambda x: 1 + x[0])
            FixedTemperatureBC(subdomain=my_subdomain,
                                 value=lambda t: 1 + t)
            FixedTemperatureBC(subdomain=my_subdomain,
                                 value=lambda x, t: 1 + x[0] + t)

    """

    def __init__(
        self,
        subdomain: _subdomain.SurfaceSubdomain,
        value: np.ndarray | fem.Constant | int | float | Callable,
    ):
        super().__init__(subdomain, value)

        if self.value.temperature_dependent:
            raise ValueError(
                "Temperature dependent boundary conditions are not supported for FixedTemperatureBC"
            )
