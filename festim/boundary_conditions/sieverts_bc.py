import festim as F
from dolfinx import fem
import ufl


def siverts_law(T, S_0, E_S, pressure):
    """Applies the Sieverts law to compute the concentration at the boundary"""
    S = S_0 * ufl.exp(-E_S / F.k_B / T)
    return S * pressure**0.5


class SievertsBC(F.DirichletBC):
    """
    Dirichlet boundary condition class

    Args:
        subdomain (festim.Subdomain): the subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition
        species (str): the name of the species

    attributes:
        subdomain (festim.Subdomain): the subdomain where the boundary
            condition is applied
        value (float or fem.Constant): the value of the boundary condition
        species (str): the name of the species
    """

    def __init__(self, subdomain, S_0, E_S, pressure, species) -> None:
        super().__init__(value=None, species=species, subdomain=subdomain)
        self.subdomain = subdomain
        self.S_0 = S_0
        self.E_S = E_S
        self.pressure = pressure
        self.species = species

    def create_formulation(self, mesh, temperature, dofs, function_space):
        """Evaluates the concentration at the boundary using the sieverts law,
        then creating the forulation

        Args:
            mesh (dolfinx.mesh.Mesh): the domain mesh
            temperature (fem.Constant): the temperature of the surface
            dofs (numpy.ndarray): the degrees of freedom of surface facets
            function_space (dolfinx.fem.FunctionSpace): the function space
        """
        value = siverts_law(
            T=temperature,
            S_0=F.as_fenics_constant(mesh=mesh, value=self.S_0),
            E_S=F.as_fenics_constant(mesh=mesh, value=self.E_S),
            pressure=F.as_fenics_constant(mesh=mesh, value=self.pressure),
        )

        form = fem.dirichletbc(
            F.as_fenics_constant(mesh=mesh, value=float(value)), dofs, function_space
        )

        return form
