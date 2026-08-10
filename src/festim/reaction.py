import warnings
from collections.abc import Callable
from functools import reduce
from operator import mul

import ufl
from dolfinx import fem

from festim import k_B
from festim.helpers import Value
from festim.source import ParticleSource
from festim.species import ImplicitSpecies, Species
from festim.subdomain.volume_subdomain import VolumeSubdomain


class ReactionBase:
    """A reaction that produces or consumes species at an arbitrary net rate,
    taking place within a volume.

    The net rate is a single coefficient

    .. math::

        R = f(c_i, x, T, t)

    given by ``reaction_rate``, a festim.Value that can be a float, a ufl
    expression, or a callable of the temperature (argument ``T``), the spatial
    coordinate (``x``), the time (``t``) and/or other species concentrations
    (referenced by name through ``arg_to_species``). ``R`` is whatever
    ``reaction_rate`` returns, so rates such as :math:`R = k (c_1 - c_2)` are
    expressible.

    A reaction does not enter the formulation directly: it is expanded into
    volumetric :class:`~festim.source.ParticleSource` objects (see
    :meth:`create_sources`), each reactant getting a sink ``-R`` and each product
    a source ``+R``. The stoichiometry is carried by the ``reactant`` and
    ``product`` lists themselves: a species listed twice as a reactant is
    consumed at rate ``2 R``.

    Arguments:
        reaction_rate: The net reaction rate coefficient :math:`R`.
        volume: The volume subdomain where the reaction takes place.
        reactant: The reactant(s), consumed at rate ``R`` each. Implicit species
            have no governing equation and receive no source.
        product: The product(s), produced at rate ``R`` each. ``None`` for a
            reaction with no product.
        arg_to_species: A dictionary mapping argument names in a callable
            ``reaction_rate`` to festim.Species objects, allowing the rate to
            depend on species concentrations. Every argument of ``reaction_rate``
            other than the reserved ``t``/``x``/``T`` must appear as a key; extra
            keys that the rate does not use are ignored with a warning. Every value
            must be a festim.Species. Alternatively the mapping may be attached
            directly to a ``reaction_rate`` passed as a festim.Value (its
            ``species_dependent_value``), but the two ways are mutually exclusive:
            giving a mapping both here and on the rate Value raises a ValueError.

    Attributes:
        reaction_rate: The net reaction rate coefficient, as a festim.Value.
        volume: The volume subdomain where the reaction takes place.
        reactant: The reactant(s), as a list.
        product: The product(s), as a list (empty for a reaction with no product).
        arg_to_species: The mapping used to resolve concentration arguments in a
            callable reaction rate.

    Examples:

        .. testcode:: ReactionBase

            import festim as F

            material = F.Material(D_0=1, E_D=0)
            volume = F.VolumeSubdomain(id=1, material=material)

            A = F.Species("A")
            B = F.Species("B")

            # an arbitrary rate R = 2 * (c_A - c_B): A is consumed and B produced
            reaction = F.ReactionBase(
                reaction_rate=lambda c_A, c_B: 2.0 * (c_A - c_B),
                volume=volume,
                reactant=A,
                product=B,
                arg_to_species={"c_A": A, "c_B": B},
            )
            print(reaction)

        .. testoutput:: ReactionBase

            A --> B
    """

    volume: VolumeSubdomain
    reactant: list[Species | ImplicitSpecies]
    product: list[Species]
    reaction_rate: (
        float
        | int
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
        reaction_rate: float
        | int
        | Callable
        | fem.Constant
        | fem.Expression
        | ufl.core.expr.Expr
        | fem.Function
        | Value,
        volume: VolumeSubdomain,
        reactant: Species | ImplicitSpecies | list[Species | ImplicitSpecies],
        product: Species | list[Species] | None = None,
        arg_to_species: dict[str, Species] | None = None,
    ) -> None:
        self.volume = volume
        self.reactant = reactant
        self.product = product
        self.reaction_rate = reaction_rate
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
        # stored as a list (empty for a reaction with no product)
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
    def reaction_rate(self):
        return self._reaction_rate

    @reaction_rate.setter
    def reaction_rate(self, value):
        self._reaction_rate = value if isinstance(value, Value) else Value(value)

    @property
    def rate_coefficients(self) -> list[Value]:
        """The festim.Value rate coefficients that must be converted to fenics
        objects and updated in time. Subclasses with a different rate structure
        (e.g. :class:`GenericReaction`) override this."""
        return [self.reaction_rate]

    @property
    def species(self) -> list[Species | ImplicitSpecies]:
        """All species involved in the reaction: those receiving a source term and
        those the rate depends on. Used e.g. to update implicit-species densities."""
        involved = []
        for spe in self.reactant + self.product:
            if spe not in involved:
                involved.append(spe)
        for spe in self.arg_to_species.values():
            if spe not in involved:
                involved.append(spe)
        return involved

    @property
    def _rate_has_own_species_map(self) -> bool:
        """True if a species mapping is attached directly to a rate coefficient
        Value (its species_dependent_value). Computed live from the rates so it
        can never desync from them."""
        return any(rate.species_dependent_value for rate in self.rate_coefficients)

    @property
    def arg_to_species(self):
        return self._arg_to_species

    @arg_to_species.setter
    def arg_to_species(self, value):
        value = value or {}
        # the species mapping must be given in exactly one place
        if self._rate_has_own_species_map:
            if value:
                raise ValueError(
                    "a species mapping was given both to a rate coefficient (via the "
                    "species_dependent_value of a festim.Value) and to arg_to_species; "
                    "provide it in only one place"
                )
            # the rate Values already carry their own mapping; just expose the
            # combined mapping for introspection
            combined = {}
            for rate in self.rate_coefficients:
                combined.update(rate.species_dependent_value)
            self._arg_to_species = combined
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
        for rate in self.rate_coefficients:
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

        # 2. a key used by no rate coefficient is harmless (festim.Value ignores
        # arguments its callable does not declare), so only warn about it
        unused = set(value) - species_args
        if unused:
            warnings.warn(
                f"arg_to_species contains {sorted(unused)}, which is not an "
                "argument of any reaction rate coefficient; it will be ignored",
                stacklevel=2,
            )
        self._arg_to_species = value

        for rate in self.rate_coefficients:
            rate.species_dependent_value = value

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
        return f"{type(self).__name__}({self}, {self.reaction_rate})"

    def __str__(self) -> str:
        reactants = " + ".join(str(spe) for spe in self.reactant)
        products = " + ".join(str(spe) for spe in self.product)
        return f"{reactants} --> {products}".rstrip()

    def reaction_term(self) -> ufl.core.expr.Expr:
        """The net reaction rate ``R`` as a ufl expression.

        ``reaction_rate`` must already be converted to a fenics object (done by
        ``HydrogenTransportProblem.convert_reaction_rates_to_fenics_objects``).

        Returns:
            The net reaction rate to be unpacked into particle sources.
        """

        assert self.reaction_rate.fenics_object is not None, (
            "fenics_object has not been defined"
        )

        return self.reaction_rate.fenics_object

    def create_sources(self) -> list[ParticleSource]:
        """Express the reaction as a list of volumetric particle sources, one per
        appearance of a :class:`~festim.species.Species` in ``reactant`` or
        ``product``.

        Each reactant is given a sink ``-R`` and each product a source ``+R``,
        where ``R`` is :meth:`reaction_term`. The stoichiometry therefore comes
        from the lists themselves: a species listed twice gets two terms, which
        add up in the formulation. Implicit species (which have no governing
        equation) are skipped. The rate coefficients must already be converted to
        fenics objects.

        Returns:
            A list of festim.ParticleSource objects.
        """
        rate = self.reaction_term()
        sources = [
            ParticleSource(value=-rate, volume=self.volume, species=spe)
            for spe in self.reactant
            if isinstance(spe, Species)
        ]
        sources += [
            ParticleSource(value=rate, volume=self.volume, species=spe)
            for spe in self.product
        ]
        return sources


class GenericReaction(ReactionBase):
    """A reaction between one or more reactant species and zero or more product
    species, following a mass-action net rate

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
    :class:`~festim.species.Species` (see :meth:`create_sources`), each
    appearance in ``reactant`` consuming the species at rate ``R`` and each
    appearance in ``product`` producing it at rate ``R``.

    Arguments:
        reactant: The reactant(s). A species listed twice appears squared in the
            mass-action rate and is consumed at rate ``2 R`` (eg. ``2 A --> B``).
        product: The product(s), following the same rule. ``None`` for an
            irreversible reaction with no product.
        forward_rate: The forward reaction rate coefficient :math:`k_1`.
        volume: The volume subdomain where the reaction takes place.
        backward_rate: The backward reaction rate coefficient :math:`k_2`. If
            None, the reaction is irreversible.
        arg_to_species: A dictionary mapping argument names in a callable rate
            coefficient to festim.Species objects, allowing a coefficient to
            depend on species concentrations. Every argument of the forward and
            backward coefficients other than the reserved ``t``/``x``/``T`` must
            appear as a key; extra keys that no coefficient uses are ignored with a
            warning. Every value must be a festim.Species.
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
        | Callable
        | fem.Constant
        | fem.Expression
        | ufl.core.expr.Expr
        | fem.Function
        | Value,
        backward_rate: float
        | int
        | Callable
        | fem.Constant
        | fem.Expression
        | ufl.core.expr.Expr
        | fem.Function
        | Value
        | None = None,
        arg_to_species: dict[str, Species] | None = None,
    ) -> None:
        # the rates are set before the base __init__ because arg_to_species
        # validation inspects them
        self.forward_rate = forward_rate
        self.backward_rate = backward_rate

        super().__init__(
            reaction_rate=None,
            volume=volume,
            reactant=reactant,
            product=product,
            arg_to_species=arg_to_species,
        )

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
    def rate_coefficients(self) -> list[Value]:
        return [self.forward_rate, self.backward_rate]

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

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self}, {self.forward_rate}, {self.backward_rate})"
        )

    def __str__(self) -> str:
        # a double arrow when the backward term is actually there
        if self.backward_rate.input_value is None or not self.product:
            return super().__str__()
        reactants = " + ".join(str(spe) for spe in self.reactant)
        products = " + ".join(str(spe) for spe in self.product)
        return f"{reactants} <--> {products}"


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

    volume: VolumeSubdomain
    reactant: list[Species | ImplicitSpecies]
    product: list[Species]
    k_0: float
    E_k: float
    p_0: float | None
    E_p: float | None

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

        super().__init__(
            reactant=reactant,
            product=product,
            forward_rate=None,
            backward_rate=None,
            volume=volume,
        )
        self.p_0 = p_0
        self.E_p = E_p

    @property
    def forward_rate(self):
        if self._forward_rate is None:
            self._forward_rate = Value(
                lambda T: self.k_0 * ufl.exp(-self.E_k / (k_B * T))
            )
        return self._forward_rate

    @forward_rate.setter
    def forward_rate(self, value):
        self._forward_rate = None

    @property
    def backward_rate(self):
        if self._backward_rate is None:
            if self.product:
                self._backward_rate = Value(
                    lambda T: (
                        self.p_0 * ufl.exp(-self.E_p / (k_B * T))
                        if self.E_p
                        else self.p_0
                    )
                )
            else:
                # no product means no backward rate (a Value wrapping None)
                self._backward_rate = Value(None)
        return self._backward_rate

    @backward_rate.setter
    def backward_rate(self, value):
        # derived from p_0/E_p; assignment only invalidates the cached Value
        self._backward_rate = None

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
        return f"{type(self).__name__}({self}, {self.k_0}, {self.E_k}, {self.p_0}, {self.E_p})"  # noqa: E501


class DecayReaction(GenericReaction):
    """A first-order radioactive decay reaction, consuming a single reactant species
    at a rate set by its decay half-life.

    The net rate follows the mass-action form of :class:`GenericReaction`

    .. math::

        R = \\lambda c

    where the decay constant :math:`\\lambda = \\ln(2) / t_{1/2}` is built from the
    ``half_life`` :math:`t_{1/2}`. The reaction is irreversible (no backward rate):
    the reactant is consumed at rate ``R`` and each product, if any, is produced at
    rate ``R`` (e.g. helium from the decay of tritium).

    Arguments:
        reactant: The decaying reactant species. Exactly one species decays: a decay
            is first order, so a list of more than one reactant is rejected.
        half_life: The decay half-life, in the simulation's time unit. Must be a
            positive float.
        volume: The volume subdomain where the decay takes place.
        product: The decay product(s). ``None`` if the products are not tracked.

    Attributes:
        reactant: The reactant, as a list of one species.
        half_life: The decay half-life.
        product: The product(s), as a list (empty if none are tracked).
        forward_rate: The decay constant :math:`\\lambda`, as a festim.Value.
        volume: The volume subdomain where the decay takes place.

    Examples:

        .. testcode:: DecayReaction

            import festim as F

            material = F.Material(D_0=1, E_D=0)
            volume = F.VolumeSubdomain(id=1, material=material)

            T = F.Species("T")  # tritium
            He = F.Species("He")  # helium-3 produced by the decay

            # tritium decays into helium with a half-life of ~12.3 years (in seconds)
            reaction = F.DecayReaction(
                reactant=T,
                half_life=3.888e8,
                volume=volume,
                product=He,
            )
            print(reaction)

        .. testoutput:: DecayReaction

            T --> He
    """

    volume: VolumeSubdomain
    reactant: list[Species | ImplicitSpecies]
    product: Species | list[Species] | None
    half_life: float

    def __init__(
        self,
        reactant: Species | ImplicitSpecies,
        half_life: float,
        volume: VolumeSubdomain,
        product: Species | list[Species] | None = None,
    ) -> None:
        self.half_life = half_life

        super().__init__(
            reactant=reactant,
            product=product,
            forward_rate=None,
            backward_rate=None,
            volume=volume,
        )

    # only the setter differs from the base class, the getter is inherited
    @GenericReaction.reactant.setter
    def reactant(self, value):
        if isinstance(value, list) and len(value) > 1:
            raise ValueError(
                "reactant must be a single species for a DecayReaction, not "
                f"{len(value)}; a decay is first order in the decaying species"
            )
        ReactionBase.reactant.fset(self, value)

    @property
    def forward_rate(self):
        if self._forward_rate is None:
            self._forward_rate = Value(ufl.ln(2) / self.half_life)
        return self._forward_rate

    @forward_rate.setter
    def forward_rate(self, value):
        self._forward_rate = None

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
        self._forward_rate = None

    def __repr__(self) -> str:
        reactants = " + ".join(str(spe) for spe in self.reactant)
        return f"{type(self).__name__}({reactants}, half_life={self.half_life})"


class Reaction(ArrheniusReaction):
    """Deprecated alias for :class:`ArrheniusReaction`."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "Reaction is deprecated, use ArrheniusReaction instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
