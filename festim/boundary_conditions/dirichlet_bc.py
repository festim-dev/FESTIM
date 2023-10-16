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
        self.value_fenics, expr = F.convert_to_appropriate_obj(
            object=self.value, function_space=function_space, mesh=mesh
        )
        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            if "t" in arguments:
                if "x" in arguments:
                    self.time_dependent_expressions.append(expr)
                else:
                    self.time_dependent_expressions.append(self.value_fenics)

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
        """Updates the boundary condition value

        Args:
            t (float): the time
        """
        for expr in self.time_dependent_expressions:
            expr.t = t

        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            if "t" in arguments and "x" in arguments:
                self.value_fenics.interpolate(self.bc_expr.__call__)
            elif "t" in arguments:
                self.value_fenics.value = self.value(t=t)
