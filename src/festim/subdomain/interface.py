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
    """How to couple interfaces for discontinuous problems."""

    nitsche = 10
    penalty = 20

    @classmethod
    def from_string(cls, s: str):
        """Can be removed with Python 3.11+."""
        s = s.lower()
        if s == "nitsche":
            return cls.nitsche
        elif s == "penalty":
            return cls.penalty
        else:
            raise ValueError("interface_method must be one of 'nitsche' or 'penalty'")


class InterfaceBase(ABC):
    def __init__(
        self,
        id: int,
        subdomains: list[VolumeSubdomain],
        penalty_term: float = 10.0,
    ):
        """Class representing an interface between two subdomains.

        Args:
            id (int): the tag of the interface subdomain in the parent meshtags
            subdomains (list[F.VolumeSubdomain]): the subdomains sharing this interface
            penalty_term (float, optional): Penalty term in the Nitsche DG formulation.
                Needs to be "sufficiently large". Defaults to 10.0.
        """
        self.id = id
        self.subdomains = tuple(subdomains)
        self.penalty_term = penalty_term

    def pad_parent_maps(self):
        """Workaround to make sparsity-pattern work without skips"""
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
        """
        Compute integration data for interface integrals.
        We define the first domain on an interface as the "+" restriction,
        meaning that we must sort all integration entities in this order

        Returns
            integration_data: Integration data for interior facets
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

    @abstractmethod
    def get_formulation(
        self,
        dInterface: ufl.Measure,  # NOTE should this be called dS?
        method: InterfaceMethod,
        species: list["Species"],
        temperature,
    ) -> tuple[ufl.Form, ufl.Form]:
        pass


class Interface(InterfaceBase):
    id: int
    subdomains: tuple[VolumeSubdomain, VolumeSubdomain]
    parent_mesh: dolfinx.mesh.Mesh
    mt: dolfinx.mesh.MeshTags
    restriction: list[str, str] = ("+", "-")
    padded: bool

    # TODO this should be a method of a subclass of Interface since we want to support
    # other types of interfaces in the future
    def get_formulation(
        self,
        dInterface: ufl.Measure,  # NOTE should this be called dS?
        method: InterfaceMethod,
        species: list["Species"],
        temperature,
    ) -> tuple[ufl.Form, ufl.Form]:
        """
        Generates the interface formulation for all `species` and store the forms in
        the `.F` attribute of the subdomains.

        Args:
            dInterface: the measure corresponding to the interface, with the correct
                integration data
            method: the method to use to enforce the interface conditions
            species: the species for which the interface conditions should be applied.
                Must be defined in both subdomains of the interface.
            temperature: the temperature for the interface conditions

        Raises:
            ValueError: if the interface method is unknown
        """

        subdomain_0, subdomain_1 = self.subdomains
        res = self.restriction
        mesh = dInterface.ufl_domain()

        for spe in species:
            assert subdomain_0 in spe.subdomains and subdomain_1 in spe.subdomains, (
                f"Species {spe.name} must be defined in both subdomains of the "
                "interface for the interface conditions to be applied"
            )
            v_0 = spe.subdomain_to_test_function[subdomain_0](res[0])
            v_1 = spe.subdomain_to_test_function[subdomain_1](res[1])

            u_0 = spe.subdomain_to_solution[subdomain_0](res[0])
            u_1 = spe.subdomain_to_solution[subdomain_1](res[1])

            K_0 = subdomain_0.material.get_solubility_coefficient(
                mesh, temperature(res[0]), spe
            )
            K_1 = subdomain_1.material.get_solubility_coefficient(
                mesh, temperature(res[1]), spe
            )

            method_to_function = {
                InterfaceMethod.penalty: self.penalty_method,
                InterfaceMethod.nitsche: self.nitsche_method,
            }
            try:
                F_0, F_1 = method_to_function[method](
                    dInterface, (u_0, u_1), (K_0, K_1), (v_0, v_1)
                )
                return F_0, F_1
            except KeyError:
                raise ValueError(f"Unknown interface method {method}")

    def penalty_method(self, dInterface, us, Ks, vs):
        subdomain_0, subdomain_1 = self.subdomains
        u_0, u_1 = us
        v_0, v_1 = vs
        K_0, K_1 = Ks
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

        F_0 = self.penalty_term * ufl.inner(equality, v_0) * dInterface(self.id)
        F_1 = -self.penalty_term * ufl.inner(equality, v_1) * dInterface(self.id)

        return F_0, F_1

    def nitsche_method(self, dInterface, us, Ks, vs):
        u_0, u_1 = us
        K_0, K_1 = Ks
        v_0, v_1 = vs

        def mixed_term(u, v, n):
            return ufl.dot(ufl.grad(u), n) * v

        res = self.restriction
        n = ufl.FacetNormal(dInterface.ufl_domain())
        cr = ufl.Circumradius(dInterface.ufl_domain())
        n_0 = n(res[0])
        h_0 = 2 * cr(res[0])
        h_1 = 2 * cr(res[1])
        gamma = self.penalty_term
        F_0 = -0.5 * mixed_term((u_0 + u_1), v_0, n_0) * dInterface(
            self.id
        ) - 0.5 * mixed_term(v_0, (u_0 / K_0 - u_1 / K_1), n_0) * dInterface(self.id)

        F_1 = +0.5 * mixed_term((u_0 + u_1), v_1, n_0) * dInterface(
            self.id
        ) - 0.5 * mixed_term(v_1, (u_0 / K_0 - u_1 / K_1), n_0) * dInterface(self.id)
        F_0 += (
            2
            * gamma
            / (h_0 + h_1)
            * (u_0 / K_0 - u_1 / K_1)
            * v_0
            * dInterface(self.id)
        )
        F_1 += (
            -2
            * gamma
            / (h_0 + h_1)
            * (u_0 / K_0 - u_1 / K_1)
            * v_1
            * dInterface(self.id)
        )

        return F_0, F_1
