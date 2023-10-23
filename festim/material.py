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
                D_0={"Species_1": 1.9e-7, "Species_2": 2.0e-7},
                E_D={"Species_1": 0.2, "Species_2": 0.3},
                name="my_mat"
            )
    """

    def __init__(self, D_0, E_D, name=None) -> None:
        self.D_0 = D_0
        self.E_D = E_D
        self.name = name

    def get_diffusion_coefficient(self, mesh, temperature, species, model_species):
        """Defines the diffusion coefficient
        Args:

            mesh (dolfinx.mesh.Mesh): the domain mesh
            temperature (dolfinx.fem.Constant): the temperature
            species (festim.Species): the species
            model_species (list): the list of species in the model
        Returns:
            ufl.algebra.Product: the diffusion coefficient
        """
        if species not in model_species:
            raise ValueError(f"Species {species} not found in model species")

        if isinstance(self.D_0, (float, int)) and isinstance(self.E_D, (float, int)):
            D_0 = F.as_fenics_constant(self.D_0, mesh)
            E_D = F.as_fenics_constant(self.E_D, mesh)

            return D_0 * ufl.exp(-E_D / F.k_B / temperature)

        elif isinstance(self.D_0, dict) and isinstance(self.E_D, dict):
            # check D_0 and E_D have the same keys
            if list(self.D_0.keys()) != list(self.E_D.keys()):
                raise ValueError("D_0 and E_D have different keys")

            for key in self.D_0.keys():
                if isinstance(key, str):
                    F.find_species_from_name(key, model_species)
                elif key not in model_species:
                    raise ValueError(f"Species {key} not found in model species")

            try:
                D_0 = F.as_fenics_constant(self.D_0[species.name], mesh)
            except KeyError:
                D_0 = F.as_fenics_constant(self.D_0[species], mesh)

            try:
                E_D = F.as_fenics_constant(self.E_D[species.name], mesh)
            except KeyError:
                E_D = F.as_fenics_constant(self.E_D[species], mesh)

            return D_0 * ufl.exp(-E_D / F.k_B / temperature)

        else:
            raise ValueError("D_0 and E_D must be either floats or dicts")
