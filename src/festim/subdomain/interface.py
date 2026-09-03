from enum import Enum
from typing import TYPE_CHECKING

import dolfinx
import numpy as np
import ufl
from dolfinx.cpp.fem import compute_integration_domains
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


def compute_one_sided_interior_facet_data(
    cell_tags: "dolfinx.mesh.MeshTags",
    facets,
    subdomain: VolumeSubdomain,
):
    """Integration data for the facets of a manifold that touch ``subdomain``, ordered
    so that ``"+"`` is always ``subdomain``.

    :func:`compute_ordered_interior_facet_data` orders the two sides of facets that all
    separate the *same* pair of subdomains. That is not enough for a manifold adjacent
    to more than two volumes -- a grain-boundary network in a polycrystal where every
    grain is its own subdomain -- whose facets separate a different pair from one grain
    to the next. There, one integral per adjacent grain is used instead of one per
    manifold: this selects the facets that grain lies on and puts it on ``"+"``, so
    every coupling term is written against a single restriction.

    A facet between grains ``i`` and ``j`` is returned by both calls, once with ``i`` on
    ``"+"`` and once with ``j``, which is what lets each side carry its own exchange
    law. A facet with the same subdomain on both sides is returned unswapped.

    Args:
        cell_tags: the cell meshtags of the parent mesh
        facets: the facets of the manifold, as returned by ``MeshTags.find``
        subdomain: the volume subdomain to place on the ``"+"`` restriction

    Returns:
        a flat array of ``(cell_plus, local_facet_plus, cell_minus, local_facet_minus)``
        quadruples, the form accepted by ``ufl.Measure("dS", subdomain_data=...)``
    """
    topology = cell_tags.topology
    topology.create_connectivity(topology.dim - 1, topology.dim)

    # MeshTags.topology is already the C++ object compute_integration_domains wants
    data = compute_integration_domains(
        dolfinx.fem.IntegralType.interior_facet, topology, facets
    ).reshape(-1, 4)

    # cell index -> volume subdomain id. MeshTags are not necessarily ordered by cell,
    # nor do they necessarily cover every cell, so indexing values directly would be
    # wrong for a mesh whose cells are not all tagged in order
    cell_map = topology.index_map(topology.dim)
    lookup = np.full(cell_map.size_local + cell_map.num_ghosts, -1, dtype=np.int32)
    lookup[cell_tags.indices] = cell_tags.values

    sides = lookup[data[:, [0, 2]]]
    on_plus, on_minus = sides[:, 0] == subdomain.id, sides[:, 1] == subdomain.id
    data = data[on_plus | on_minus]
    # only the facets that have subdomain on "-" alone need their sides swapped
    swap = on_minus[on_plus | on_minus] & ~on_plus[on_plus | on_minus]
    data[swap] = data[swap][:, [2, 3, 0, 1]]
    return data.reshape(-1)


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

    def Ds(self, species: "Species", temperature):
        """Get diffusion coefficients for both sides of the interface.

        Args:
            species: The species for which to compute diffusivity.
            temperature: A function that returns temperature at given restrictions.

        Returns:
            Diffusion coefficients (D_0, D_1) for subdomains 0 and 1.
        """
        return tuple(
            subdomain.material.get_diffusion_coefficient(
                self.parent_mesh, temperature(self.restriction[i]), species
            )
            for i, subdomain in enumerate(self.subdomains)
        )

    def equality(self, species: "Species", temperature):
        """The interface constraint, as a residual that vanishes at equilibrium.

        Both sides are expressed in the same quantity so that the difference is
        meaningful: the partial pressure when the two materials obey different
        solubility laws (``c/K`` for Henry, ``(c/K)**2`` for Sievert), and plainly
        ``c/K`` when they obey the same one -- for a matching pair the squared and
        unsquared constraints have the same non-negative roots, so the linear form
        is preferred as it keeps the coupling linear.

        Note that ``penalty_term`` therefore carries different units in the two
        cases, and its values are not comparable across law pairs.

        Args:
            species: The species for which to compute the constraint.
            temperature: A function that returns temperature at given restrictions.

        Returns:
            The constraint residual, zero when the two sides are in equilibrium.

        Raises:
            ValueError: If either material has an unsupported solubility law.
        """
        subdomain_0, subdomain_1 = self.subdomains
        u_0, u_1 = self.us(species)
        K_0, K_1 = self.Ks(species, temperature)

        if subdomain_0.material.solubility_law == subdomain_1.material.solubility_law:
            return u_0 / K_0 - u_1 / K_1

        def partial_pressure(subdomain, u, K):
            match subdomain.material.solubility_law:
                case SolubilityLaw.HENRY:
                    return u / K
                case SolubilityLaw.SIEVERT:
                    return (u / K) ** 2
                case _:
                    raise ValueError(
                        "Unsupported material law "
                        + f"{subdomain.material.solubility_law}"
                    )

        return partial_pressure(subdomain_0, u_0, K_0) - partial_pressure(
            subdomain_1, u_1, K_1
        )

    def equality_scale(self, species: "Species", temperature):
        """The factor that converts :meth:`equality` into concentration units.

        ``equality`` is written in potential units -- ``c/K`` for a matching pair,
        a partial pressure for a mixed one -- so it cannot be compared to a flux
        directly. Nitsche's stabilisation and adjoint terms need it in concentration
        units, so that ``penalty_term * D / h * scale * equality`` is a flux and
        ``penalty_term`` is the dimensionless O(10) stabilisation parameter Nitsche's
        theory calls for, whatever units the problem is posed in.

        For a matching pair the scale is the mean solubility, which recovers the
        textbook jump ``c_0 - c_1`` when the two materials share a solubility. For a
        Sievert/Henry pair it is the Henry coefficient, exactly ``dc/dP`` on that
        side, so ``scale * equality`` reads as the concentration the Henry side is
        missing relative to equilibrium -- polynomial in both unknowns, with none of
        the degeneracy of the Sievert side's ``dc/dP = K**2/(2c)`` at ``c = 0``.

        Args:
            species: The species for which to compute the scale.
            temperature: A function that returns temperature at given restrictions.

        Returns:
            A factor with units of concentration over ``equality``'s units.
        """
        subdomain_0, subdomain_1 = self.subdomains
        K_0, K_1 = self.Ks(species, temperature)

        if subdomain_0.material.solubility_law == subdomain_1.material.solubility_law:
            return 0.5 * (K_0 + K_1)
        if subdomain_0.material.solubility_law == SolubilityLaw.HENRY:
            return K_0
        return K_1

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

        The interface is modelled as a surface exchange: the same flux
        ``penalty_term * equality`` leaves one side and enters the other, so the
        chemical potential drops across the interface by ``flux / penalty_term``
        and particles are conserved exactly at any ``penalty_term``.

        Args:
            dS: Integration measure for the interface.
            species: The species for which to compute the interface form.
            temperature: A function returning temperature at given restrictions.

        Returns:
            Variational forms for subdomains 0 and 1.
        """
        v_0, v_1 = self.vs(species)
        equality = self.equality(species, temperature)

        F_0 = self.penalty_term * ufl.inner(equality, v_0) * dS(self.id)
        F_1 = -self.penalty_term * ufl.inner(equality, v_1) * dS(self.id)

        return F_0, F_1

    def nitsche_method(self, dS, species, temperature):
        """Generate interface formulation using the Nitsche method.

        Nitsche's method adds, on top of the penalty stabilisation, the term that
        makes the formulation *consistent*: the flux the two sides must agree on,
        ``{D grad(c) . n}``, appears explicitly, so the exact solution satisfies the
        discrete form for any ``penalty_term``. The penalty then only has to make the
        system stable rather than to enforce the interface condition on its own,
        which is why a value of order 10-100 reaches an accuracy the pure penalty
        needs orders of magnitude more for.

        The symmetric (adjoint-consistent) variant is used. Like the penalty, both
        the consistency and the stabilisation term enter the two sides equally and
        oppositely, so particles are conserved exactly whatever ``equality`` is; and
        like the penalty it goes through :meth:`equality`, so a Sievert/Henry pair is
        coupled through partial pressures rather than through ``c/K``. Unlike the
        penalty, ``penalty_term`` is dimensionless here: the constraint is brought
        into concentration units by :meth:`equality_scale` first, so the same value
        of order 10 works whatever units the problem is posed in.

        Args:
            dS: Integration measure for the interface.
            species: The species for which to compute the interface form.
            temperature: A function returning temperature at given restrictions.

        Returns:
            Variational forms for subdomains 0 and 1.
        """
        u_0, u_1 = self.us(species)
        v_0, v_1 = self.vs(species)
        D_0, D_1 = self.Ds(species, temperature)
        # in concentration units, so that the two terms below are fluxes and
        # penalty_term stays a dimensionless stabilisation parameter
        jump = self.equality_scale(species, temperature) * self.equality(
            species, temperature
        )

        res = self.restriction
        n_0 = ufl.FacetNormal(dS.ufl_domain())(res[0])
        cr = ufl.Circumradius(dS.ufl_domain())
        h_0 = 2 * cr(res[0])
        h_1 = 2 * cr(res[1])

        def flux(u, D):
            """The diffusive flux of ``u`` through the interface, along n_0."""
            return D * ufl.dot(ufl.grad(u), n_0)

        # {D grad(u) . n}: at the exact solution both sides equal the transmitted
        # flux, so this term reproduces it and the two below vanish
        avg_flux = 0.5 * (flux(u_0, D_0) + flux(u_1, D_1))
        # gamma * D / h : turns the concentration jump into a flux
        stabilisation = self.penalty_term * (D_0 + D_1) / (h_0 + h_1)

        # consistency
        F_0 = -avg_flux * v_0 * dS(self.id)
        F_1 = +avg_flux * v_1 * dS(self.id)
        # adjoint consistency (symmetric variant)
        F_0 += -0.5 * flux(v_0, D_0) * jump * dS(self.id)
        F_1 += -0.5 * flux(v_1, D_1) * jump * dS(self.id)
        # stabilisation
        F_0 += stabilisation * jump * v_0 * dS(self.id)
        F_1 += -stabilisation * jump * v_1 * dS(self.id)

        return F_0, F_1
