import warnings

from ufl import exp
from ufl.core.expr import Expr

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
    forward_rate: Value
    backward_rate: Value
    arg_to_species: dict[str, Species]

    def __init__(
        self,
        volume: VolumeSubdomain,
        reactant: Species | ImplicitSpecies | list[Species | ImplicitSpecies],
        product: Species | list[Species] | None,
        forward_rate: Value,
        backward_rate=None,
        arg_to_species: dict[str, Species] | None = None,
    ) -> None:
        self.volume = volume
        self.reactant = reactant
        self.product = product
        self.forward_rate = forward_rate
        self.backward_rate = backward_rate
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
        self._forward_rate = Value(input_value=value)

    @property
    def backward_rate(self):
        return self._backward_rate

    @backward_rate.setter
    def backward_rate(self, value):
        self._backward_rate = Value(input_value=value)

    @property
    def arg_to_species(self):
        return self._arg_to_species

    @arg_to_species.setter
    def arg_to_species(self, value):
        value = value or {}
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

    def _evaluate_rate(self, rate: Value, temperature, function_space, subdomain):
        """Convert a rate coefficient festim.Value to a fenics object at the given
        temperature. Returns 0 for an absent (irreversible) backward rate."""
        if rate.input_value is None:
            return 0
        rate.convert_input_value(
            function_space=function_space,
            temperature=temperature,
            subdomain=subdomain,
            up_to_ufl_expr=True,
        )
        return rate.fenics_object

    def reaction_term(
        self,
        temperature,
        function_space=None,
        subdomain: VolumeSubdomain | None = None,
        reactant_concentrations: list | None = None,
        product_concentrations: list | None = None,
    ) -> Expr:
        """Compute the net reaction rate ``R`` as a ufl expression.

        Arguments:
            temperature: The temperature at which the rate is computed.
            function_space: The function space used to convert the rate
                coefficients, needed when a coefficient is a float or depends on
                the spatial coordinate.
            subdomain: The volume subdomain on which concentrations are
                evaluated, needed in the discontinuous case.
            reactant_concentrations: The concentrations of the reactants. Must
                be the same length as the reactants. If None, the
                ``concentration`` attribute of each reactant is used. If an
                element is None, the ``concentration`` attribute of that
                reactant is used.
            product_concentrations: The concentrations of the products, with the
                same rules as ``reactant_concentrations``.

        Returns:
            The net reaction rate to be used in a formulation.
        """
        products = self.product

        # detect if mixed_domain
        mixed_domain = any(
            isinstance(reactant, Species) and reactant.subdomain_to_solution != {}
            for reactant in self.reactant
        ) or any(
            isinstance(product, Species) and product.subdomain_to_solution != {}
            for product in products
        )

        def get_concentration(species):
            if mixed_domain:
                return species.concentration_submesh(self.volume)
            return species.concentration

        # reaction rate coefficients
        k = self._evaluate_rate(
            self.forward_rate, temperature, function_space, subdomain
        )
        p = self._evaluate_rate(
            self.backward_rate, temperature, function_space, subdomain
        )

        # if reactant_concentrations is provided, use these concentrations
        reactants = self.reactant
        if reactant_concentrations is not None:
            assert len(reactant_concentrations) == len(reactants)
            for i, reactant in enumerate(reactants):
                if reactant_concentrations[i] is None:
                    reactant_concentrations[i] = get_concentration(reactant)
        else:
            reactant_concentrations = [
                get_concentration(reactant) for reactant in reactants
            ]

        # if product_concentrations is provided, use these concentrations
        if product_concentrations is not None:
            assert len(product_concentrations) == len(products)
            for i, product in enumerate(products):
                if product_concentrations[i] is None:
                    product_concentrations[i] = get_concentration(product)
        else:
            product_concentrations = [
                get_concentration(product) for product in products
            ]

        # multiply all concentrations to be used in the term
        product_of_reactants = reactant_concentrations[0]
        for reactant_conc in reactant_concentrations[1:]:
            product_of_reactants *= reactant_conc

        if products:
            product_of_products = product_concentrations[0]
            for product_conc in product_concentrations[1:]:
                product_of_products *= product_conc
        else:
            product_of_products = 0

        return (k * product_of_reactants) - (p * product_of_products)

    def create_sources(
        self,
        temperature,
        function_space=None,
        subdomain: VolumeSubdomain | None = None,
    ) -> list[ParticleSource]:
        """Express the reaction as a list of volumetric particle sources, one per
        participating :class:`~festim.species.Species`.

        Each reactant is consumed (source value ``-R``) and each product is
        produced (source value ``+R``), where ``R`` is :meth:`reaction_term`.
        Implicit species (which have no governing equation) are skipped.

        Arguments:
            temperature: The temperature at which the rate is computed.
            function_space: The function space used to convert the rate
                coefficients.
            subdomain: The volume subdomain on which concentrations are
                evaluated, needed in the discontinuous case.

        Returns:
            A list of festim.ParticleSource objects.
        """
        rate = self.reaction_term(temperature, function_space, subdomain)

        sources = [
            ParticleSource(value=-rate, volume=self.volume, species=reactant)
            for reactant in self.reactant
            if isinstance(reactant, Species)
        ]
        sources += [
            ParticleSource(value=rate, volume=self.volume, species=product)
            for product in self.product
            if isinstance(product, Species)
        ]
        return sources


class ArrheniusReaction(GenericReaction):
    """A reaction between species, with forward and backward rate coefficients
    built from Arrhenius laws. This is typically used to model
    trapping/detrapping.

    Args:
        reactant (Union[F.Species, F.ImplicitSpecies], List[Union[F.Species,
            F.ImplicitSpecies]]): The reactant.
        product (Optional[Union[F.Species, List[F.Species]]]): The product.
        k_0 (float): The forward rate constant pre-exponential factor.
        E_k (float): The forward rate constant activation energy.
        p_0 (float): The backward rate constant pre-exponential factor.
        E_p (float): The backward rate constant activation energy.
        volume (F.VolumeSubdomain): The volume subdomain where the reaction
            takes place.

    Attributes:
        reactant (Union[F.Species, F.ImplicitSpecies], List[Union[F.Species,
            F.ImplicitSpecies]]): The reactant.
        product (Optional[Union[F.Species, List[F.Species]]]): The product.
        k_0 (float): The forward rate constant pre-exponential factor.
        E_k (float): The forward rate constant activation energy.
        p_0 (float): The backward rate constant pre-exponential factor.
        E_p (float): The backward rate constant activation energy.
        volume (F.VolumeSubdomain): The volume subdomain where the reaction
            takes place.

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
        self.p_0 = p_0
        self.E_p = E_p

        def forward_rate(T):
            return self.k_0 * exp(-self.E_k / (_k_B * T))

        backward_rate = None
        if self.p_0 is not None:

            def backward_rate(T):
                if self.p_0 and self.E_p:
                    return self.p_0 * exp(-self.E_p / (_k_B * T))
                elif self.p_0:
                    return self.p_0
                else:
                    return 0

        super().__init__(
            reactant=reactant,
            product=product,
            forward_rate=forward_rate,
            backward_rate=backward_rate,
            volume=volume,
        )

    def reaction_term(
        self,
        temperature,
        function_space=None,
        subdomain: VolumeSubdomain | None = None,
        reactant_concentrations: list | None = None,
        product_concentrations: list | None = None,
    ) -> Expr:
        # validate that p_0/E_p are consistent with the presence of products
        # only at this point (not at __init__) so that Reaction objects can be
        # built ahead of time even if p_0/E_p aren't set yet
        if not self.product:
            if self.p_0 is not None:
                raise ValueError(
                    f"p_0 must be None, not {self.p_0}"
                    + " when no products are present."
                )
            if self.E_p is not None:
                raise ValueError(
                    f"E_p must be None, not {self.E_p}"
                    + " when no products are present."
                )
        else:
            if self.p_0 is None:
                raise ValueError(
                    "p_0 cannot be None when reaction products are present."
                )
            elif self.E_p is None:
                raise ValueError(
                    "E_p cannot be None when reaction products are present."
                )

        return super().reaction_term(
            temperature,
            function_space,
            subdomain,
            reactant_concentrations,
            product_concentrations,
        )

    def __repr__(self) -> str:
        reactants = " + ".join([str(reactant) for reactant in self.reactant])
        products = " + ".join([str(product) for product in self.product])
        return f"{type(self).__name__}({reactants} <--> {products}, {self.k_0}, {self.E_k}, {self.p_0}, {self.E_p})"  # noqa: E501


class Reaction(ArrheniusReaction):
    """Deprecated alias for :class:`ArrheniusReaction`."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "Reaction is deprecated, use ArrheniusReaction instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
