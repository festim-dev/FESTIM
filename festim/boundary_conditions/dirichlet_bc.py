import festim as F
from dolfinx import fem


class DirichletBC:
    """
    Dirichlet boundary condition class

    Args:
        subdomain (festim.Subdomain): the surface subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition
        species (str): the name of the species

    attributes:
        subdomain (festim.Subdomain): the surface subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition
        species (str): the name of the species
    """

    def __init__(self, subdomain, value, species) -> None:
        self.subdomain = subdomain
        self.value = value
        self.species = species

    def define_surface_subdominan_dofs(self, facet_meshtags, mesh, function_space):
        """Defines the facets and the degrees of freedom of the boundary
        condition

        Args:
            mesh (dolfinx.mesh.Mesh): the domain mesh
        """
        bc_facets = facet_meshtags.find(self.subdomain.id)
        bc_dofs = fem.locate_dofs_topological(function_space, mesh.fdim, bc_facets)
        return bc_dofs

    def create_formulation(self, mesh, dofs, function_space, temperature):
        """Applies the boundary condition
        Args:
            mesh (dolfinx.mesh.Mesh): the domain mesh
            value (float or fem.Constant): the value of the boundary condition
            dofs (numpy.ndarray): the degrees of freedom of surface facets
            function_space (dolfinx.fem.FunctionSpace): the function space
        """
        form = fem.dirichletbc(
            F.as_fenics_constant(mesh=mesh, value=self.value), dofs, function_space
        )
        return form
