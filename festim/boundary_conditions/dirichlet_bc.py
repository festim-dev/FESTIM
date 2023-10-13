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

        self.form = None

    def create_formulation(self, mesh, value, dofs, function_space):
        """Applies the boundary condition
        Args:
            mesh (dolfinx.mesh.Mesh): the domain mesh
            value (float or fem.Constant): the value of the boundary condition
            dofs (numpy.ndarray): the degrees of freedom of surface facets
            function_space (dolfinx.fem.FunctionSpace): the function space
        """
        if not isinstance(value, (float, int)):
            raise TypeError(
                f"Boundary condition value must be float or int, not {type(value)}"
            )

        self.form = fem.dirichletbc(
            F.as_fenics_constant(mesh, value), dofs, function_space
        )
