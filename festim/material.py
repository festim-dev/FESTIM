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

    Usage:
        >>> my_mat = Material(D_0=1.9e-7, E_D=0.2, name="my_mat")
    or if several species:
        >>> my_mat = Material(
                D_0={"Species_1": 1.9e-7, "Species_2": 2.0e-7},
                E_D={"Species_1": 0.2, "Species_2": 0.3},
                name="my_mat"
            )
    """

    def __init__(self, D_0, E_D, name=None) -> None:
        self.D_0 = D_0
        self.E_D = E_D
        self.name = name

    def get_D_0(self, species=None):
        """Returns the pre-exponential factor of the diffusion coefficient
        Args:
            species (festim.Species or str, optional): the species we want the pre-exponential
                factor of the diffusion coefficient of. Only needed if D_0 is a dict.
        Returns:
            float: the pre-exponential factor of the diffusion coefficient
        """

        if isinstance(self.D_0, (float, int)):
            return self.D_0

        elif isinstance(self.D_0, dict):
            if species is None:
                raise ValueError("species must be provided if D_0 is a dict")

            if species in self.D_0:
                return self.D_0[species]
            elif species.name in self.D_0:
                return self.D_0[species.name]
            else:
                raise ValueError(f"{species} is not in D_0 keys")

        else:
            raise ValueError("D_0 must be either a float or a dict")

    def get_E_D(self, species=None):
        """Returns the activation energy of the diffusion coefficient
        Args:
            species (festim.Species or str, optional): the species we want the activation
                energy of the diffusion coefficient of. Only needed if E_D is a dict.
        Returns:
            float: the activation energy of the diffusion coefficient
        """

        if isinstance(self.E_D, (float, int)):
            return self.E_D

        elif isinstance(self.E_D, dict):
            if species is None:
                raise ValueError("species must be provided if E_D is a dict")

            if species in self.E_D:
                return self.E_D[species]
            elif species.name in self.E_D:
                return self.E_D[species.name]
            else:
                raise ValueError(f"{species} is not in E_D keys")

        else:
            raise ValueError("E_D must be either a float or a dict")

    def get_diffusion_coefficient(self, mesh, temperature, species=None):
        """Defines the diffusion coefficient
        Args:

            mesh (dolfinx.mesh.Mesh): the domain mesh
            temperature (dolfinx.fem.Constant): the temperature
            species (festim.Species, optional): the species we want the diffusion
                coefficient of. Only needed if D_0 and E_D are dicts.
        Returns:
            ufl.algebra.Product: the diffusion coefficient
        """
        # TODO use get_D_0 and get_E_D to refactore this method, something like:
        # D_0 = self.get_D_0(species=species)
        # E_D = self.get_E_D(species=species)

        # D_0 = F.as_fenics_constant(D_0, mesh)
        # E_D = F.as_fenics_constant(E_D, mesh)

        # return D_0 * ufl.exp(-E_D / F.k_B / temperature)

        if isinstance(self.D_0, (float, int)) and isinstance(self.E_D, (float, int)):
            D_0 = F.as_fenics_constant(self.D_0, mesh)
            E_D = F.as_fenics_constant(self.E_D, mesh)

            return D_0 * ufl.exp(-E_D / F.k_B / temperature)

        elif isinstance(self.D_0, dict) and isinstance(self.E_D, dict):
            # check D_0 and E_D have the same keys
            # this check should go in a setter
            if list(self.D_0.keys()) != list(self.E_D.keys()):
                raise ValueError("D_0 and E_D have different keys")

            if species is None:
                raise ValueError("species must be provided if D_0 and E_D are dicts")

            if species in self.D_0:
                D_0 = F.as_fenics_constant(self.D_0[species], mesh)
            elif species.name in self.D_0:
                D_0 = F.as_fenics_constant(self.D_0[species.name], mesh)
            else:
                raise ValueError(f"{species} is not in D_0 keys")

            if species in self.E_D:
                E_D = F.as_fenics_constant(self.E_D[species], mesh)
            elif species.name in self.E_D:
                E_D = F.as_fenics_constant(self.E_D[species.name], mesh)

            return D_0 * ufl.exp(-E_D / F.k_B / temperature)

        else:
            raise ValueError("D_0 and E_D must be either floats or dicts")
