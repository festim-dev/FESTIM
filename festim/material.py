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
            mesh (dolfinx.mesh.Mesh): the domain mesh
            temperature (dolfinx.fem.Constant): the temperature
        Returns:
            ufl.algebra.Product: the diffusion coefficient
        """

        # check type of values and convert to fem.Constant if needed
        if isinstance(self.D_0, (float, int)):
            self.D_0 = fem.Constant(mesh, float(self.D_0))
        elif isinstance(self.D_0, fem.Constant):
            pass
        else:
            raise TypeError(
                f"D_0 must be float, int or dolfinx.fem.Constant, not {type(self.D_0)}"
            )
        if isinstance(self.E_D, (float, int)):
            self.E_D = fem.Constant(mesh, float(self.E_D))
        elif isinstance(self.E_D, fem.Constant):
            pass
        else:
            raise TypeError(
                f"E_D must be float, int or dolfinx.fem.Constant, not {type(self.E_D)}"
            )

        return self.D_0 * ufl.exp(-self.E_D / F.k_B / temperature)
