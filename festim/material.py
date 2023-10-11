import ufl
from dolfinx import fem
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

    def define_diffusion_coefficient(self, mesh, temperature):
        """Defines the diffusion coefficient
        Args:
            temperature (dolfinx.fem.Constant): the temperature
        Returns:
            float: the diffusion coefficient
        """

        if isinstance(self.D_0, float):
            self.D_0 = fem.Constant(mesh, self.D_0)
        if isinstance(self.E_D, float):
            self.E_D = fem.Constant(mesh, self.E_D)

        return self.D_0 * ufl.exp(-self.E_D / F.k_B / temperature)
