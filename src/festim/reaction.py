import warnings
from collections.abc import Callable
from functools import reduce
from operator import mul

import numpy as np
import ufl.core.expr
from dolfinx import fem
from ufl import exp

from festim import k_B as _k_B
from festim.helpers import Value
from festim.source import ParticleSource
from festim.species import ImplicitSpecies, Species
from festim.subdomain.volume_subdomain import VolumeSubdomain


class GenericReaction:
    """A generic reaction between one or more reactant species and zero or more
    product species, taking place within a volume.

    The reaction follows a mass-action net rate

    .. math::

        R = k_1 \\prod_i c_i^{\\text{reactant}} - k_2 \\prod_j c_j^{\\text{product}}

    where :math:`k_1` and :math:`k_2` are the ``forward_rate`` and
    ``backward_rate`` coefficients. Both coefficients are festim.Value objects,
    so they can be a float, a ufl expression, or a callable of the temperature
    (argument ``T``) and/or of other species concentrations (referenced by name
    through ``arg_to_species``). This paves the way for building other
    reaction types (eg. hydride formation) on top of this class.

    A reaction does not enter the formulation directly: it is expanded into one
    volumetric :class:`~festim.source.ParticleSource` per participating
    :class:`~festim.species.Species` (see :meth:`create_sources`), each reactant
    being consumed at rate ``R`` and each product produced at rate ``R``.

    Arguments:
        reactant: The reactant(s).
        product: The product(s). ``None`` for an irreversible reaction with no
            product.
        forward_rate: The forward reaction rate coefficient :math:`k_1`.
        volume: The volume subdomain where the reaction takes place.
        backward_rate: The backward reaction rate coefficient :math:`k_2`. If
            None, the reaction is irreversible.
        arg_to_species: A dictionary mapping argument names in a callable rate
            coefficient to festim.Species objects, allowing a coefficient to
            depend on species concentrations. It must map exactly the arguments
            of the forward and backward coefficients other than the reserved
            ``t``/``x``/``T``: every such argument must appear as a key, and every
            key must be such an argument. Every value must be a festim.Species.
            Alternatively the mapping may be attached directly to a rate
            coefficient passed as a festim.Value (its ``species_dependent_value``),
            but the two ways are mutually exclusive: giving a mapping both here and
            on a rate Value raises a ValueError.

    Attributes:
        reactant: The reactant(s).
        product: The product(s), as a list (empty for an irreversible reaction).
        forward_rate: The forward reaction rate coefficient, as a festim.Value.
        backward_rate: The backward reaction rate coefficient, as a festim.Value.
        volume: The volume subdomain where the reaction takes place.
        arg_to_species: The mapping used to resolve concentration
            arguments in a callable rate coefficient.

    Examples:

        .. testcode:: GenericReaction

            import festim as F

            material = F.Material(D_0=1, E_D=0)
            volume = F.VolumeSubdomain(id=1, material=material)

            A = F.Species("A")
            B = F.Species("B")
            C = F.Species("C")

            # net rate R = k1 * c_A * c_B - k2 * c_C, with k1 depending on T
            reaction = F.GenericReaction(
                reactant=[A, B],
                product=C,
                forward_rate=lambda T: 1e-3 * T,
                backward_rate=2.0,
                volume=volume,
            )
            print(reaction)

        .. testoutput:: GenericReaction

            A + B <--> C
    """

    volume: VolumeSubdomain
    reactant: list[Species | ImplicitSpecies]
    product: list[Species]
    forward_rate: (
        float
        | int
        | np.ndarray
        | Callable
        | fem.Constant
        | fem.Expression
        | ufl.core.expr.Expr
        | fem.Function
        | Value
    )
    backward_rate: (
        float
        | int
        | np.ndarray
        | Callable
        | fem.Constant
        | fem.Expression
        | ufl.core.expr.Expr
        | fem.Function
        | Value
    )
    arg_to_species: dict[str, Species]

    def __init__(
        self,
        volume: VolumeSubdomain,
        reactant: Species | ImplicitSpecies | list[Species | ImplicitSpecies],
        product: Species | list[Species] | None,
        forward_rate: float
        | int
        | np.ndarray
        | Callable
        | fem.Constant
        | fem.Expression
        | ufl.core.expr.Expr
        | fem.Function
        | Value,
        backward_rate: float
        | int
        | np.ndarray
        | Callable
        | fem.Constant
        | fem.Expression
        | ufl.core.expr.Expr
        | fem.Function
        | Value
        | None = None,
        arg_to_species: dict[str, Species] | None = None,
    ) -> None:
        self.volume = volume
        self.reactant = reactant
        self.product = product
        self.forward_rate = forward_rate
        self.backward_rate = backward_rate
        # a species mapping may be attached directly to a rate coefficient Value
        # (its species_dependent_value); remember that so the arg_to_species setter
        # can refuse to receive the same information twice
        self._rate_has_own_species_map = bool(
            self.forward_rate.species_dependent_value
            or self.backward_rate.species_dependent_value
        )
        self.arg_to_species = arg_to_species

    @property
    def reactant(self):
        return self._reactant

    @reactant.setter
    def reactant(self, value):
        if not isinstance(value, list):
            value = [value]
        if len(value) == 0:
            raise ValueError(
                "reactant must be an entry of one or more species objects, not an empty list."  # noqa: E501
            )
        for i in value:
            if not isinstance(i, (Species, ImplicitSpecies)):
                raise TypeError(
                    "reactant must be an F.Species or F.ImplicitSpecies, not "
                    + f"{type(i)}"
                )
        self._reactant = value

    @property
    def product(self):
        return self._product

    @product.setter
    def product(self, value):
        # stored as a list (empty for an irreversible reaction with no product)
        if value is None:
            value = []
        elif not isinstance(value, list):
            value = [value]
        for i in value:
            if not isinstance(i, Species):
                raise TypeError(
                    "product must be an F.Species, a list of F.Species, or None, "
                    + f"not {type(i).__name__}"
                )
        self._product = value

    @property
    def forward_rate(self):
        return self._forward_rate

    @forward_rate.setter
    def forward_rate(self, value):
        self._forward_rate = value if isinstance(value, Value) else Value(value)

    @property
    def backward_rate(self):
        return self._backward_rate

    @backward_rate.setter
    def backward_rate(self, value):
        self._backward_rate = value if isinstance(value, Value) else Value(value)

    @property
    def arg_to_species(self):
        return self._arg_to_species

    @arg_to_species.setter
    def arg_to_species(self, value):
        value = value or {}
        # the species mapping must be given in exactly one place
        if getattr(self, "_rate_has_own_species_map", False):
            if value:
                raise ValueError(
                    "a species mapping was given both to a rate coefficient (via the "
                    "species_dependent_value of a festim.Value) and to arg_to_species; "
                    "provide it in only one place"
                )
            # the rate Values already carry their own mapping; just expose the
            # combined mapping for introspection
            self._arg_to_species = {
                **self.forward_rate.species_dependent_value,
                **self.backward_rate.species_dependent_value,
            }
            return
        for name, spe in value.items():
            if not isinstance(spe, Species):
                raise TypeError(
                    "arg_to_species values must be a festim.Species, not "
                    + f"{type(spe).__name__} (for key {name!r})"
                )
        # the species-dependent arguments of the rate coefficients are their
        # callable arguments other than the reserved t/x/T handled by festim.Value
        reserved = {"t", "x", "T"}
        species_args = set()
        for rate in (self.forward_rate, self.backward_rate):
            if callable(rate.input_value):
                code = rate.input_value.__code__
                species_args |= set(code.co_varnames[: code.co_argcount]) - reserved
        # 1. a rate coefficient must not depend on a species missing from the mapping
        unmapped = species_args - set(value)
        if unmapped:
            raise ValueError(
                f"the reaction rate coefficients depend on {sorted(unmapped)}, "
                "which must be given in arg_to_species"
            )
        # 2. the mapping must not contain a key used by no rate coefficient
        unused = set(value) - species_args
        if unused:
            raise ValueError(
                f"arg_to_species contains {sorted(unused)}, which is not an "
                "argument of any reaction rate coefficient"
            )
        self._arg_to_species = value
        # hand the same mapping to each coefficient; each festim.Value ignores the
        # keys its own callable does not declare (see helpers.as_mapped_function)
        self.forward_rate.species_dependent_value = value
        self.backward_rate.species_dependent_value = value

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        if not isinstance(value, VolumeSubdomain):
            raise TypeError(
                f"volume must be a festim.VolumeSubdomain, not {type(value).__name__}"
            )
        self._volume = value

    def __repr__(self) -> str:
        reactants = " + ".join([str(reactant) for reactant in self.reactant])
        products = " + ".join([str(product) for product in self.product])
        return (
            f"{type(self).__name__}({reactants} <--> {products}, "
            f"{self.forward_rate}, {self.backward_rate})"
        )

    def __str__(self) -> str:
        reactants = " + ".join([str(reactant) for reactant in self.reactant])
        products = " + ".join([str(product) for product in self.product])
        return f"{reactants} <--> {products}"

    @property
    def _mixed_domain(self) -> bool:
        """True in the discontinuous case, where each species has one solution per
        subdomain rather than a single concentration attribute.

        NOTE: becomes always True once the discontinuous formulation is the only
        problem path; this property (and the branch on it below) can be dropped."""
        return any(
            isinstance(spe, Species) and spe.subdomain_to_solution != {}
            for spe in self.reactant + self.product
        )

    def _concentrations(self, species: list, overrides: list | None) -> list:
        """The concentration of each species. Where a (non-None) override is given
        for a species it is used in place of the species' own concentration."""
        # NOTE: `overrides` exists only for the change-of-variable problem
        # (HydrogenTransportProblemDiscontinuousChangeVar, its u*K_S substitution)
        # and can be removed together with the reaction_term arguments feeding it
        # once that class is dropped
        overrides = overrides or [None] * len(species)
        concentrations = []
        for spe, override in zip(species, overrides, strict=True):
            if override is not None:
                concentrations.append(override)
            elif self._mixed_domain:
                concentrations.append(spe.concentration_submesh(self.volume))
            else:
                concentrations.append(spe.concentration)
        return concentrations

    def reaction_term(
        self,
        # NOTE: reactant_concentrations/product_concentrations exist only for the
        # change-of-variable problem and can be removed with it (see _concentrations)
        reactant_concentrations: list | None = None,
        product_concentrations: list | None = None,
    ) -> ufl.core.expr.Expr:
        """Compute the net reaction rate ``R`` as a ufl expression.

        The rate coefficients must already be converted to fenics objects (done by
        ``HydrogenTransportProblem.convert_reaction_rates_to_fenics_objects``).

        Arguments:
            reactant_concentrations: The concentration to use for each reactant,
                same length as the reactants. Where an entry is None the reactant's
                own concentration is used. If None, all reactant concentrations
                are used.
            product_concentrations: The concentrations of the products, following
                the same rules as ``reactant_concentrations``.

        Returns:
            The net reaction rate to be used in a formulation.
        """
        reactants = self._concentrations(self.reactant, reactant_concentrations)
        forward = self.forward_rate.fenics_object * reduce(mul, reactants)

        # no backward term for an irreversible reaction or one with no products
        if self.backward_rate.input_value is None or not self.product:
            return forward

        products = self._concentrations(self.product, product_concentrations)
        backward = self.backward_rate.fenics_object * reduce(mul, products)
        return forward - backward

    def create_sources(self) -> list[ParticleSource]:
        """Express the reaction as a list of volumetric particle sources, one per
        participating :class:`~festim.species.Species`.

        Each reactant is consumed (source value ``-R``) and each product is
        produced (source value ``+R``), where ``R`` is :meth:`reaction_term`.
        Implicit species (which have no governing equation) are skipped. The rate
        coefficients must already be converted to fenics objects.

        Returns:
            A list of festim.ParticleSource objects.
        """
        rate = self.reaction_term()

        sources = [
            ParticleSource(value=-rate, volume=self.volume, species=reactant)
            for reactant in self.reactant
            if isinstance(reactant, Species)
        ]
        sources += [
            ParticleSource(value=rate, volume=self.volume, species=product)
            for product in self.product
        ]
        return sources


class ArrheniusReaction(GenericReaction):
    """A reaction between species, with forward and backward rate coefficients
    built from Arrhenius laws. This is typically used to model
    trapping/detrapping.

    Arguments:
        reactant: The reactant(s).
        product: The product(s). None for an irreversible reaction with no
            product.
        k_0: The forward rate constant pre-exponential factor.
        E_k: The forward rate constant activation energy.
        volume: The volume subdomain where the reaction takes place.
        p_0: The backward rate constant pre-exponential factor. Must be None when
            there is no product.
        E_p: The backward rate constant activation energy. Must be None when there
            is no product.

    Attributes:
        reactant: The reactant(s).
        product: The product(s), as a list (empty for an irreversible reaction).
        k_0: The forward rate constant pre-exponential factor.
        E_k: The forward rate constant activation energy.
        p_0: The backward rate constant pre-exponential factor.
        E_p: The backward rate constant activation energy.
        volume: The volume subdomain where the reaction takes place.

    Examples:

        .. testcode:: ArrheniusReaction

            import festim as F

            # create a volume subdomain for the reaction to take place in
            material = F.Material(D_0=1, E_D=0)
            volume = F.VolumeSubdomain(id=1, material=material)

            # create two reactant species and a product species
            reactant = [F.Species("A"), F.Species("B")]
            product = F.Species("C")

            # create a reaction between the reactants and the product
            reaction = F.ArrheniusReaction(
                reactant=reactant,
                product=product,
                k_0=1.0,
                E_k=0.2,
                p_0=0.1,
                E_p=0.3,
                volume=volume,
            )
            print(reaction)

        .. testoutput:: ArrheniusReaction

            A + B <--> C
    """

    def __init__(
        self,
        reactant: Species | ImplicitSpecies | list[Species | ImplicitSpecies],
        k_0: float,
        E_k: float,
        volume: VolumeSubdomain,
        product: Species | list[Species] | None = None,
        p_0: float | None = None,
        E_p: float | None = None,
    ) -> None:
        self.k_0 = k_0
        self.E_k = E_k

        def forward_rate(T):
            return self.k_0 * exp(-self.E_k / (_k_B * T))

        def backward_rate(T):
            if self.E_p:
                return self.p_0 * exp(-self.E_p / (_k_B * T))
            return self.p_0

        super().__init__(
            reactant=reactant,
            product=product,
            forward_rate=forward_rate,
            backward_rate=backward_rate if p_0 is not None else None,
            volume=volume,
        )
        # set after product so the setters can check the two are consistent
        self.p_0 = p_0
        self.E_p = E_p

    @property
    def p_0(self):
        return self._p_0

    @p_0.setter
    def p_0(self, value):
        if not self.product and value is not None:
            raise ValueError(
                f"p_0 must be None, not {value} when no products are present."
            )
        if self.product and value is None:
            raise ValueError("p_0 cannot be None when reaction products are present.")
        self._p_0 = value

    @property
    def E_p(self):
        return self._E_p

    @E_p.setter
    def E_p(self, value):
        if not self.product and value is not None:
            raise ValueError(
                f"E_p must be None, not {value} when no products are present."
            )
        if self.product and value is None:
            raise ValueError("E_p cannot be None when reaction products are present.")
        self._E_p = value

    def __repr__(self) -> str:
        reactants = " + ".join([str(reactant) for reactant in self.reactant])
        products = " + ".join([str(product) for product in self.product])
        return f"{type(self).__name__}({reactants} <--> {products}, {self.k_0}, {self.E_k}, {self.p_0}, {self.E_p})"  # noqa: E501


class DecayReaction(GenericReaction):
    """A first-order radioactive decay reaction, consuming one or more reactant
    species at a rate set by their decay half-life.

    The net rate follows the mass-action form of :class:`GenericReaction`

    .. math::

        R = \\lambda \\prod_i c_i

    where the decay constant :math:`\\lambda = \\ln(2) / t_{1/2}` is built from the
    ``half_life`` :math:`t_{1/2}`. The reaction is irreversible: it has no product
    and no backward rate, and each reactant is consumed at rate ``R`` (a sink).

    Arguments:
        reactant: The decaying reactant species.
        half_life: The decay half-life, in the same time unit as the simulation.
            Must be a positive float.
        volume: The volume subdomain where the decay takes place.

    Attributes:
        reactant: The reactant(s).
        half_life: The decay half-life.
        forward_rate: The decay constant :math:`\\lambda`, as a festim.Value.
        volume: The volume subdomain where the decay takes place.

    Examples:

        .. testcode:: DecayReaction

            import festim as F

            material = F.Material(D_0=1, E_D=0)
            volume = F.VolumeSubdomain(id=1, material=material)

            T = F.Species("T")  # tritium

            # tritium decays with a half-life of ~12.3 years (in seconds)
            reaction = F.DecayReaction(
                reactant=T,
                half_life=3.888e8,
                volume=volume,
            )
            print(reaction)

        .. testoutput:: DecayReaction

            T -->
    """

    def __init__(
        self,
        reactant: Species | ImplicitSpecies | list[Species | ImplicitSpecies],
        half_life: float,
        volume: VolumeSubdomain,
    ) -> None:
        # forward_rate is a placeholder here: the half_life setter (called below)
        # derives the decay constant and assigns it. Radioactive decay is
        # irreversible, hence no product and no backward rate.
        super().__init__(
            reactant=reactant,
            product=None,
            forward_rate=0.0,
            backward_rate=None,
            volume=volume,
        )
        self.half_life = half_life

    @property
    def half_life(self):
        return self._half_life

    @half_life.setter
    def half_life(self, value):
        if not isinstance(value, float | int) or isinstance(value, bool):
            raise TypeError(f"half_life must be a float, not {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"half_life must be positive, not {value}")
        self._half_life = value
        # decay constant lambda = ln(2) / t_1/2, kept in sync with half_life
        self.forward_rate = np.log(2) / value

    def __repr__(self) -> str:
        reactants = " + ".join([str(reactant) for reactant in self.reactant])
        return f"{type(self).__name__}({reactants}, half_life={self.half_life})"

    def __str__(self) -> str:
        reactants = " + ".join([str(reactant) for reactant in self.reactant])
        return f"{reactants} -->"


class Reaction(ArrheniusReaction):
    """Deprecated alias for :class:`ArrheniusReaction`."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "Reaction is deprecated, use ArrheniusReaction instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
