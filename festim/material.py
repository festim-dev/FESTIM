import ufl
import festim as F


class Material:
    """
    Material class
    """

    def __init__(self, D_0, E_D, name=None) -> None:
        """Inits Material
        Args:
            D_0 (float or fem.Constant): the diffusion coefficient at 0 K
            E_D (float or fem.Constant): the activation energy for diffusion
            name (str): the name of the material
        """
        self.D_0 = D_0
        self.E_D = E_D
        self.name = name

    def get_diffusion_coefficient(self, mesh, temperature):
        """Defines the diffusion coefficient
        Args:
            mesh (dolfinx.mesh.Mesh): the domain mesh
            temperature (dolfinx.fem.Constant): the temperature
        Returns:
            ufl.algebra.Product: the diffusion coefficient
        """

        # check type of values
        D_0 = F.as_fenics_constant(self.D_0, mesh)
        E_D = F.as_fenics_constant(self.E_D, mesh)

        return D_0 * ufl.exp(-E_D / F.k_B / temperature)
