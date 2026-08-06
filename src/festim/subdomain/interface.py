from enum import Enum
from typing import TYPE_CHECKING

import dolfinx
import ufl
from scifem.mesh import compute_interface_data

from festim.material import SolubilityLaw
from festim.subdomain.volume_subdomain import VolumeSubdomain

if TYPE_CHECKING:
    from festim.species import Species

from abc import ABC, abstractmethod


def compute_ordered_interior_facet_data(
    cell_tags: "dolfinx.mesh.MeshTags",
    facet_tags: "dolfinx.mesh.MeshTags",
    tag: int,
    subdomain_plus: VolumeSubdomain,
    subdomain_minus: VolumeSubdomain,
):
    """Integration data for an interior-facet (``dS``) integral, with the restrictions
    ordered so that ``"+"`` is always ``subdomain_plus``.

    DOLFINx's own ordering of the two cells of an interior facet is arbitrary, so
    without this any expression that treats the two sides differently -- a solubility
    jump, or a codimensional coupling with different exchange rates on either side --
    would silently get its sides swapped.

    Args:
        cell_tags: the cell meshtags of the parent mesh, marking every cell adjacent
            to the tagged facets with the id of the volume subdomain it belongs to
        facet_tags: the facet meshtags of the parent mesh
        tag: the value identifying the facets to integrate over
        subdomain_plus: the volume subdomain to place on the ``"+"`` restriction
        subdomain_minus: the volume subdomain to place on the ``"-"`` restriction

    Returns:
        ``(tag, integration_data)``, the pair accepted by ``ufl.Measure("dS",
        subdomain_data=...)``. ``integration_data`` is a flat array of
        ``(cell_plus, local_facet_plus, cell_minus, local_facet_minus)`` quadruples.

    Raises:
        ValueError: if a tagged facet does not separate the two subdomains
    """
    topology = cell_tags.topology
    topology.create_connectivity(topology.dim - 1, topology.dim)

    # scifem orders the two cells of a facet by their cell tag, lowest one on "+"
    integration_data = compute_interface_data(cell_tags, facet_tags.find(tag))
    if subdomain_plus.id > subdomain_minus.id:
        integration_data = integration_data[:, [2, 3, 0, 1]]

    # A wrong ordering is silent, so rather than trust the tags, check that the two
    # cells of every facet really do lie one in each subdomain. A bare assert would
    # vanish under ``python -O``.
    sides = cell_tags.values[integration_data[:, [0, 2]]]
    if not (sides == [subdomain_plus.id, subdomain_minus.id]).all():
        raise ValueError(
            f"facets tagged {tag} do not all separate volume subdomain "
            f"{subdomain_plus.id} from volume subdomain {subdomain_minus.id}; the "
            '"+"/"-" restrictions cannot be ordered consistently'
        )

    return (tag, integration_data.reshape(-1))


class InterfaceMethod(Enum):
    """Methods for enforcing interface continuity in discontinuous problems.

    Attributes:
        nitsche: Nitsche's method - a stabilized discontinuous Galerkin approach
            that uses average gradients and penalty stabilization.
        penalty: Pure penalty method - enforces continuity through a penalty term
            scaled by the penalty_term parameter.
    """

    nitsche = 10
    penalty = 20

    @classmethod
    def from_string(cls, s: str) -> "InterfaceMethod":
        """Convert string to InterfaceMethod enum.

        Args:
            s: String representation ('nitsche' or 'penalty').

        Returns:
            InterfaceMethod: The corresponding enum value.

        Raises:
            ValueError: If string is not 'nitsche' or 'penalty'.
        """
        s = s.lower()
        if s == "nitsche":
            return cls.nitsche
        elif s == "penalty":
            return cls.penalty
        else:
            raise ValueError("interface_method must be one of 'nitsche' or 'penalty'")


class InterfaceBase(ABC):
    """Abstract base class for interfaces between subdomains.

    Provides common functionality for handling interfaces in discontinuous finite
    element problems, including integration data computation and restriction handling.
    """

    def __init__(
        self,
        id: int,
        subdomains: list[VolumeSubdomain],
    ):
        """Initialize an interface between two subdomains.

        Args:
            id: Tag of the interface subdomain in the parent mesh tags.
            subdomains: The subdomains sharing this interface.
        """
        self.id = id
        self.subdomains = tuple(subdomains)

    def compute_mapped_interior_facet_data(self, cell_tags):
        """Compute integration data for interface integrals.

        This method computes the mapping between physical facets on the interface
        and the corresponding cells in each subdomain. It ensures that restrictions
        are ordered consistently with the first subdomain on the "+" side.

        Args:
            cell_tags: The cell meshtags of the parent mesh.

        Returns:
            tuple: A tuple of (interface_id, flattened_integration_data) where
                integration_data contains the mapped cell and facet indices.
        """
        return compute_ordered_interior_facet_data(
            cell_tags, self.mt, self.id, self.subdomains[0], self.subdomains[1]
        )

    def us(self, species: "Species"):
        """Get solution fields restricted to each side of the interface.

        Args:
            species: The species for which to get solution fields.

        Returns:
            tuple: Solution fields (u_0, u_1) restricted to ("+", "-") sides.
        """
        return tuple(
            species.subdomain_to_solution[subdomain](res)
            for subdomain, res in zip(self.subdomains, self.restriction)
        )

    def vs(self, species: "Species"):
        """Get test functions restricted to each side of the interface.

        Args:
            species: The species for which to get test functions.

        Returns:
            tuple: Test functions (v_0, v_1) restricted to ("+", "-") sides.
        """
        return tuple(
            species.subdomain_to_test_function[subdomain](res)
            for subdomain, res in zip(self.subdomains, self.restriction)
        )

    @abstractmethod
    def get_formulation(
        self,
        dS: ufl.Measure,
        method: InterfaceMethod,
        species: list["Species"],
        temperature,
    ) -> tuple[ufl.Form, ufl.Form]:
        """Generate variational forms for interface conditions.

        Args:
            dS: Integration measure for the interface.
            method: The method to enforce interface conditions.
            species: List of species for which to compute interface conditions.
            temperature: Temperature field or function for temperature-dependent laws.

        Returns:
            Variational forms for each subdomain.
        """
        pass


class Interface(InterfaceBase):
    """Represents an interface between two subdomains with discontinuous solutions.

    This class handles the coupling of solutions across an interface between two
    volume subdomains using either penalty or Nitsche methods. It manages the
    exchange of boundary conditions and enforces continuity across the interface.

    Attributes:
        id: Tag of the interface subdomain in the parent mesh tags.
        subdomains: The two subdomains
            sharing this interface.
        parent_mesh: The parent mesh containing the interface.
        mt: Mesh tags for the parent mesh.
        restriction: FEniCS restriction operators for each side
            of the interface, defaults to ("+", "-").
        method: The method used to enforce interface conditions
            (penalty or Nitsche).
        penalty_term: Penalty parameter for the interface formulation.
    """

    id: int
    subdomains: tuple[VolumeSubdomain, VolumeSubdomain]
    parent_mesh: dolfinx.mesh.Mesh
    mt: dolfinx.mesh.MeshTags
    restriction: list[str, str] = ("+", "-")
    method: InterfaceMethod

    def __init__(
        self,
        id: int,
        subdomains: list[VolumeSubdomain],
        penalty_term: float = 10.0,
        method: InterfaceMethod = InterfaceMethod.penalty,
    ):
        """Initialize an interface between two subdomains.

        Args:
            id: Tag of the interface subdomain in the parent mesh tags.
            subdomains: A list of exactly two subdomains that share this interface.
            penalty_term: Penalty parameter for the interface formulation.
                Must be sufficiently large. Defaults to 10.0.
            method: The method to enforce interface conditions.
                Defaults to InterfaceMethod.penalty.
        """
        super().__init__(id, subdomains)
        self.penalty_term = penalty_term
        self.method = method

    @property
    def method(self) -> InterfaceMethod:
        """Get the interface coupling method.

        Returns:
            InterfaceMethod: The current interface method (penalty or Nitsche).
        """
        return self._method

    @method.setter
    def method(self, value: InterfaceMethod | str) -> None:
        """Set the interface coupling method.

        Args:
            value: The method to use. Can be an InterfaceMethod enum value
                or a string ('penalty' or 'nitsche').

        Raises:
            TypeError: If value is neither an InterfaceMethod nor a string.
        """
        if isinstance(value, InterfaceMethod):
            self._method = value
        elif isinstance(value, str):
            self._method = InterfaceMethod.from_string(value)
        else:
            raise TypeError("method_interface must be of type str or InterfaceMethod")

    def Ks(self, species: "Species", temperature):
        """Get solubility coefficients for both sides of the interface.

        Computes the solubility coefficient at the interface temperature for each
        subdomain's material.

        Args:
            species: The species for which to compute solubility.
            temperature: A function that returns temperature at given restrictions.

        Returns:
            Solubility coefficients (K_0, K_1) for subdomains 0 and 1.
        """
        return tuple(
            subdomain.material.get_solubility_coefficient(
                self.parent_mesh, temperature(self.restriction[i]), species
            )
            for i, subdomain in enumerate(self.subdomains)
        )

    def get_formulation(
        self,
        dS: ufl.Measure,
        species: list["Species"],
        temperature,
    ) -> tuple[ufl.Form, ufl.Form]:
        """Generate the interface formulation for all species.

        Args:
            dS: Integration measure for the interface, with correct integration data.
            species: Species for which interface conditions should be applied.
                Must be defined in both subdomains of the interface.
            temperature: Temperature field/function for temperature-dependent laws.

        Returns:
            Variational forms to be added to each subdomain.

        Raises:
            AssertionError: If the interface method is unknown or species is not
                defined in both subdomains.
        """

        subdomain_0, subdomain_1 = self.subdomains
        F_0, F_1 = dolfinx.fem.form(0), dolfinx.fem.form(0)
        method_to_function = {
            InterfaceMethod.penalty: self.penalty_method,
            InterfaceMethod.nitsche: self.nitsche_method,
        }
        assert self.method in method_to_function, (
            f"Unknown interface method {self.method}"
        )

        for spe in species:
            assert subdomain_0 in spe.subdomains and subdomain_1 in spe.subdomains, (
                f"Species {spe.name} must be defined in both subdomains of the "
                "interface for the interface conditions to be applied"
            )
            _F_0, _F_1 = method_to_function[self.method](dS, spe, temperature)
            F_0 += _F_0
            F_1 += _F_1

        return F_0, F_1

    def penalty_method(self, dS, species, temperature):
        """Generate interface formulation using the penalty method.

        The penalty method enforces interface continuity through a penalty term:
        penalty_term * (u_1/K_1 - u_0/K_0) applied symmetrically to both sides.
        Handles different solubility laws (Henry vs Sievert) on each side.

        Args:
            dS: Integration measure for the interface.
            species: The species for which to compute the interface form.
            temperature: A function returning temperature at given restrictions.

        Returns:
            Variational forms for subdomains 0 and 1.
        """
        subdomain_0, subdomain_1 = self.subdomains
        u_0, u_1 = self.us(species)
        v_0, v_1 = self.vs(species)
        K_0, K_1 = self.Ks(species, temperature)
        if subdomain_0.material.solubility_law == subdomain_1.material.solubility_law:
            left = u_0 / K_0
            right = u_1 / K_1
        else:
            match subdomain_0.material.solubility_law:
                case SolubilityLaw.HENRY:
                    left = u_0 / K_0
                case SolubilityLaw.SIEVERT:
                    left = (u_0 / K_0) ** 2
                case _:
                    raise ValueError(
                        "Unsupported material law "
                        + f"{subdomain_0.material.solubility_law}"
                    )

            match subdomain_1.material.solubility_law:
                case SolubilityLaw.HENRY:
                    right = u_1 / K_1
                case SolubilityLaw.SIEVERT:
                    right = (u_1 / K_1) ** 2
                case _:
                    raise ValueError(
                        f"Unsupported material law "
                        f"{subdomain_1.material.solubility_law}"
                    )

        equality = right - left

        F_0 = self.penalty_term * ufl.inner(equality, v_0) * dS(self.id)
        F_1 = -self.penalty_term * ufl.inner(equality, v_1) * dS(self.id)

        return F_0, F_1

    def nitsche_method(self, dS, species, temperature):
        """Generate interface formulation using the Nitsche method.

        The Nitsche method is a stabilized discontinuous Galerkin approach that
        enforces interface continuity through a combination of:
        - Average gradient terms
        - Jump-based penalty stabilization

        This method is more stable for certain problems compared to pure penalty.

        Args:
            dS: Integration measure for the interface.
            species: The species for which to compute the interface form.
            temperature: A function returning temperature at given restrictions.

        Returns:
            Variational forms for subdomains 0 and 1.
        """
        u_0, u_1 = self.us(species)
        K_0, K_1 = self.Ks(species, temperature)
        v_0, v_1 = self.vs(species)

        def mixed_term(u, v, n):
            return ufl.dot(ufl.grad(u), n) * v

        res = self.restriction
        n = ufl.FacetNormal(dS.ufl_domain())
        cr = ufl.Circumradius(dS.ufl_domain())
        n_0 = n(res[0])
        h_0 = 2 * cr(res[0])
        h_1 = 2 * cr(res[1])
        gamma = self.penalty_term
        F_0 = -0.5 * mixed_term((u_0 + u_1), v_0, n_0) * dS(self.id) - 0.5 * mixed_term(
            v_0, (u_0 / K_0 - u_1 / K_1), n_0
        ) * dS(self.id)

        F_1 = +0.5 * mixed_term((u_0 + u_1), v_1, n_0) * dS(self.id) - 0.5 * mixed_term(
            v_1, (u_0 / K_0 - u_1 / K_1), n_0
        ) * dS(self.id)
        F_0 += 2 * gamma / (h_0 + h_1) * (u_0 / K_0 - u_1 / K_1) * v_0 * dS(self.id)
        F_1 += -2 * gamma / (h_0 + h_1) * (u_0 / K_0 - u_1 / K_1) * v_1 * dS(self.id)

        return F_0, F_1
