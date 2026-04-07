from enum import Enum
from typing import TYPE_CHECKING

import dolfinx
import numpy as np
import ufl
from dolfinx.cpp.fem import compute_integration_domains
from packaging.version import Version

from festim.material import SolubilityLaw
from festim.subdomain.volume_subdomain import VolumeSubdomain

if TYPE_CHECKING:
    from festim.species import Species

from abc import ABC, abstractmethod


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

    Provides common functionality for handling interfaces in discontinuous
    finite element problems, including integration data computation and
    restriction handling.
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

    def pad_parent_maps(self):
        """Pad parent-to-submesh maps for correct sparsity pattern.

        This is a workaround to ensure the sparsity pattern is correct when
        assembling forms with interface integrals. It pads the mapping between
        parent mesh cells and submesh cells for DOLFINx versions that require it.
        """
        try:
            # No padding needed for latest version of DOLFINx
            from dolfinx.mesh import EntityMap  # noqa: F401

            return
        except ImportError:
            pass

        if Version(dolfinx.__version__) == Version("0.9.0"):
            args = (
                dolfinx.fem.IntegralType.interior_facet,
                self.parent_mesh.topology._cpp_object,
                self.mt.find(self.id),
                self.mt.dim,
            )
        elif Version(dolfinx.__version__) > Version("0.9.0"):
            args = (
                dolfinx.fem.IntegralType.interior_facet,
                self.parent_mesh.topology._cpp_object,
                self.mt.find(self.id),
            )

        integration_data = compute_integration_domains(*args).reshape(-1, 4)
        for i in range(2):
            # We pad the parent-to-submesh map so that the sparsity pattern
            # is correct.
            mapped_cell_0 = self.subdomains[i].parent_to_submesh[integration_data[:, 0]]
            mapped_cell_1 = self.subdomains[i].parent_to_submesh[integration_data[:, 2]]
            max_cells = np.maximum(mapped_cell_0, mapped_cell_1)
            self.subdomains[i].parent_to_submesh[integration_data[:, 0]] = max_cells
            self.subdomains[i].parent_to_submesh[integration_data[:, 2]] = max_cells
            self.subdomains[i].padded = True

    def compute_mapped_interior_facet_data(self, mesh):
        """Compute integration data for interface integrals.

        This method computes the mapping between physical facets on the interface
        and the corresponding cells in each subdomain. It ensures that restrictions
        are ordered consistently with the first subdomain on the "+" side.

        Args:
            mesh: The parent mesh.

        Returns:
            tuple: A tuple of (interface_id, flattened_integration_data) where
                integration_data contains the mapped cell and facet indices.
        """
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

        if Version(dolfinx.__version__) == Version("0.9.0"):
            args = (
                dolfinx.fem.IntegralType.interior_facet,
                self.parent_mesh.topology._cpp_object,
                self.mt.find(self.id),
                self.mt.dim,
            )
        elif Version(dolfinx.__version__) > Version("0.9.0"):
            args = (
                dolfinx.fem.IntegralType.interior_facet,
                mesh.topology._cpp_object,
                self.mt.find(self.id),
            )

        integration_data = compute_integration_domains(*args)

        ordered_integration_data = integration_data.reshape(-1, 4).copy()

        try:
            # No padding needed for latest version of DOLFINx
            from dolfinx.mesh import EntityMap  # noqa: F401

            mapped_cell_0 = self.subdomains[0].cell_map.sub_topology_to_topology(
                integration_data[0::4], inverse=True
            )
            mapped_cell_1 = self.subdomains[0].cell_map.sub_topology_to_topology(
                integration_data[2::4], inverse=True
            )
            legacy_entity_map = False

        except ImportError:
            assert (not self.subdomains[0].padded) and (not self.subdomains[1].padded)
            mapped_cell_0 = self.subdomains[0].parent_to_submesh[integration_data[0::4]]
            mapped_cell_1 = self.subdomains[0].parent_to_submesh[integration_data[2::4]]
            legacy_entity_map = True

        switch = mapped_cell_1 > mapped_cell_0
        # Order restriction on one side
        if True in switch:
            ordered_integration_data[switch, :] = ordered_integration_data[switch][
                :, [2, 3, 0, 1]
            ]

        # Check that other restriction lies in other interface
        if legacy_entity_map:
            domain1_cell = self.subdomains[1].parent_to_submesh[
                ordered_integration_data[:, 2]
            ]
        else:
            domain1_cell = self.subdomains[1].cell_map.sub_topology_to_topology(
                ordered_integration_data[:, 2], inverse=True
            )
        assert (domain1_cell >= 0).all()

        return (self.id, ordered_integration_data.reshape(-1))

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
        temperature=None,
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
        padded: Whether the parent-to-submesh maps have been padded.
        method: The method used to enforce interface conditions
            (penalty or Nitsche).
        penalty_term: Penalty parameter for the interface formulation.
    """

    id: int
    subdomains: tuple[VolumeSubdomain, VolumeSubdomain]
    parent_mesh: dolfinx.mesh.Mesh
    mt: dolfinx.mesh.MeshTags
    restriction: list[str, str] = ("+", "-")
    padded: bool
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


class ContactResistance(InterfaceBase):
    def __init__(
        self,
        id: int,
        subdomains: list[VolumeSubdomain],
        contact_resistance: float,
    ):
        self.contact_resistance = contact_resistance
        super().__init__(id, subdomains)

    def get_formulation(self, dS, species, temperature=None):
        subdomain_0, subdomain_1 = self.subdomains
        res = self.restriction
        _F_0, _F_1 = dolfinx.fem.form(0), dolfinx.fem.form(0)

        for spe in species:
            assert subdomain_0 in spe.subdomains and subdomain_1 in spe.subdomains, (
                f"Species {spe.name} must be defined in both subdomains of the "
                "interface for the interface conditions to be applied"
            )
            v_0 = spe.subdomain_to_test_function[subdomain_0](res[0])
            v_1 = spe.subdomain_to_test_function[subdomain_1](res[1])

            u_0 = spe.subdomain_to_solution[subdomain_0](res[0])
            u_1 = spe.subdomain_to_solution[subdomain_1](res[1])
            F0 = -(u_1 - u_0) / self.contact_resistance * v_0 * dS(self.id)
            F1 = (u_1 - u_0) / self.contact_resistance * v_1 * dS(self.id)
            _F_0 += F0
            _F_1 += F1

        return _F_0, _F_1
