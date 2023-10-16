import festim as F
from dolfinx import fem
import numpy as np


class DirichletBC:
    """
    Dirichlet boundary condition class

    Args:
        subdomain (festim.Subdomain): the surface subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition
        species (str): the name of the species

    Attributes:
        subdomain (festim.Subdomain): the surface subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition
        species (str): the name of the species
        value_fenics (fem.Function or fem.Constant): the value of the boundary condition in
            fenics format
    """

    def __init__(self, subdomain, value, species) -> None:
        self.subdomain = subdomain
        self.value = value
        self.species = species

        self.value_fenics = None
        self.bc_expr = None
        self.time_dependent_expressions = []

    # write setter getter for value_fenics
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

    def define_surface_subdomain_dofs(self, facet_meshtags, mesh, function_space):
        """Defines the facets and the degrees of freedom of the boundary
        condition

        Args:
            mesh (festim.Mesh): the domain mesh
        """
        bc_facets = facet_meshtags.find(self.subdomain.id)
        bc_dofs = fem.locate_dofs_topological(function_space, mesh.fdim, bc_facets)
        return bc_dofs

    def create_value(self, mesh, function_space, temperature):
        if isinstance(self.value, (int, float)):
            # case 1 constant value
            self.value_fenics = F.as_fenics_constant(mesh=mesh, value=float(self.value))
        elif callable(self.value):
            arguments = self.value.__code__.co_varnames
            if "T" in arguments:
                # TODO implement case where it's T dependent

                raise NotImplementedError(
                    "Case where the value is temperature dependent is not implemented yet"
                )
            if "t" in arguments and "x" in arguments:
                # case 2: space and time dependent bc
                self.value_fenics = fem.Function(function_space)
                self.bc_expr = F.SpaceTimeDependentExpression(function=self.value, t=0)
                self.value_fenics.interpolate(self.bc_expr.__call__)

                self.time_dependent_expressions.append(self.bc_expr)

            elif "x" in arguments:
                # case 3: space dependent bc
                self.value_fenics = fem.Function(function_space)
                self.value_fenics.interpolate(self.value)

            elif "t" in arguments:
                # case 4: time dependent bc
                self.value_fenics = F.as_fenics_constant(
                    mesh=mesh, value=float(self.value(t=0))
                )

    def create_formulation(self, dofs, function_space):
        """Applies the boundary condition
        Args:
            dofs (numpy.ndarray): the degrees of freedom of surface facets
            function_space (dolfinx.fem.FunctionSpace): the function space
        """
        if isinstance(self.value_fenics, fem.Function):
            form = fem.dirichletbc(
                value=self.value_fenics,
                dofs=dofs,
            )
        else:
            form = fem.dirichletbc(
                value=self.value_fenics,
                dofs=dofs,
                V=function_space,
            )
        return form

    def update(self, t):
        for expr in self.time_dependent_expressions:
            expr.t = t

        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            if "t" in arguments and "x" in arguments:
                self.value_fenics.interpolate(self.bc_expr.__call__)
            elif "t" in arguments:
                function = self.value
                self.value_fenics.value = function(t=t)
