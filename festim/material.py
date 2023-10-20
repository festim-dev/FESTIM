import ufl
import festim as F


class Material:
    """
    Material class

    Args:
        D_0 (float or dict): the pre-exponential factor of the
            diffusion coefficient (m2/s)
        E_D (float or dict): the activation energy of the diffusion
            coeficient (eV)
        name (str): the name of the material

    Attributes:
        D_0 (float or dict): the pre-exponential factor of the
            diffusion coefficient (m2/s)
        E_D (float or dict): the activation energy of the diffusion
            coeficient (eV)
        name (str): the name of the material

    Usage (1 species):
        >>> my_mat = Material(D_0=1.9e-7, E_D=0.2, name="my_mat")
    Usage (multispecies):
        >>> my_mat = Material(
                D_0={"Species_1": 1.9e-7, "D": 2.0e-7},
                E_D={"H": 0.2, "D": 0.3},
                name="my_mat"
            )
    """

    def __init__(self, D_0, E_D, name=None) -> None:
        self.D_0 = D_0
        self.E_D = E_D
        self.name = name

    def get_diffusion_coefficient(self, mesh, temperature, species):
        """Defines the diffusion coefficient
        Args:
            mesh (dolfinx.mesh.Mesh): the domain mesh
            temperature (dolfinx.fem.Constant): the temperature
        Returns:
            ufl.algebra.Product: the diffusion coefficient
        """
        if isinstance(self.D_0, (float, int)) and isinstance(self.E_D, (float, int)):
            D_0 = F.as_fenics_constant(self.D_0, mesh)
            E_D = F.as_fenics_constant(self.E_D, mesh)

            return D_0 * ufl.exp(-E_D / F.k_B / temperature)

        elif isinstance(self.D_0, dict) and isinstance(self.E_D, dict):
            # check lengths of dicts are the same
            if len(self.D_0) != len(self.E_D):
                raise ValueError(
                    "The number of pre-exponential factors and activation energies"
                    "must be the same"
                )
            # check D_0 keys for species or species.name
            if species in self.D_0.keys():
                D_0 = F.as_fenics_constant(self.D_0[species], mesh)
            elif species.name in self.D_0.keys():
                D_0 = F.as_fenics_constant(self.D_0[species.name], mesh)
            else:
                raise ValueError("Species not defined")

            # check E_D keys for species or species.name
            if species in self.E_D.keys():
                E_D = F.as_fenics_constant(self.E_D[species], mesh)
            elif species.name in self.E_D.keys():
                E_D = F.as_fenics_constant(self.E_D[species.name], mesh)
            else:
                raise ValueError("Species not definedy")

        return D_0 * ufl.exp(-E_D / F.k_B / temperature)
