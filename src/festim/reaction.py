from typing import Union

from ufl import exp
from ufl.core.expr import Expr

from festim import k_B as _k_B
from festim.species import ImplicitSpecies as _ImplicitSpecies
from festim.species import Species as _Species
from festim.subdomain.volume_subdomain import VolumeSubdomain1D as VS1D


class GenericReaction:
    """A generic reaction between one or more reactant species and zero or more
    product species, taking place within a volume.

    The forward and backward rates are not restricted to Arrhenius laws: they can
    be a float, a ufl expression, or a callable that takes the temperature and
    returns either of those. This makes it possible to build other reaction types
    (eg. trapping, hydride formation) on top of this class.

    Arguments:
        reactant (Union[F.Species, F.ImplicitSpecies], List[Union[F.Species,
        F.ImplicitSpecies]]): The reactant(s).
        product (Optional[Union[F.Species, List[F.Species]]]): The product(s).
        forward_rate: The forward reaction rate. Can be a float, a ufl expression,
            or a callable of temperature returning either of those.
        volume (F.VolumeSubdomain1D): The volume subdomain where the reaction
        takes place.
        backward_rate: The backward reaction rate, following the same rules as
        forward_rate. If None, the reaction is irreversible.

    Attributes:
        reactant (Union[F.Species, F.ImplicitSpecies], List[Union[F.Species,
        F.ImplicitSpecies]]): The reactant(s).
        product (Optional[Union[F.Species, List[F.Species]]]): The product(s).
        forward_rate: The forward reaction rate.
        backward_rate: The backward reaction rate.
        volume (F.VolumeSubdomain1D): The volume subdomain where the reaction
        takes place.
    """

    def __init__(
        self,
        reactant: _Species | _ImplicitSpecies | list[_Species | _ImplicitSpecies],
        product: Union[_Species, list[_Species]] | None,
        forward_rate,
        volume: VS1D,
        backward_rate=None,
    ) -> None:
        self.reactant = reactant
        self.product = product
        self.forward_rate = forward_rate
        self.backward_rate = backward_rate
        self.volume = volume

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
            if not isinstance(i, (_Species, _ImplicitSpecies)):
                raise TypeError(
                    "reactant must be an F.Species or F.ImplicitSpecies, not "
                    + f"{type(i)}"
                )
        self._reactant = value

    def __repr__(self) -> str:
        reactants = " + ".join([str(reactant) for reactant in self.reactant])
        if isinstance(self.product, list):
            products = " + ".join([str(product) for product in self.product])
        else:
            products = self.product
        return (
            f"{type(self).__name__}({reactants} <--> {products}, "
            f"{self.forward_rate}, {self.backward_rate})"
        )

    def __str__(self) -> str:
        reactants = " + ".join([str(reactant) for reactant in self.reactant])
        if isinstance(self.product, list):
            products = " + ".join([str(product) for product in self.product])
        else:
            products = self.product
        return f"{reactants} <--> {products}"

    @staticmethod
    def _evaluate_rate(rate, temperature):
        """Evaluate a rate at a given temperature, allowing rates expressed as
        floats, ufl expressions, or callables of temperature."""
        return rate(temperature) if callable(rate) else rate

    def reaction_term(
        self,
        temperature,
        reactant_concentrations: list | None = None,
        product_concentrations: list | None = None,
    ) -> Expr:
        """Compute the reaction term at a given temperature.

        Arguments:
            temperature: The temperature at which the reaction term is computed.
            reactant_concentrations: The concentrations of the reactants. Must
                be the same length as the reactants. If None, the ``concentration``
                attribute of each reactant is used. If an element is None, the
                ``concentration`` attribute of the reactant is used.
            product_concentrations: The concentrations of the products. Must
                be the same length as the products. If None, the ``concentration``
                attribute of each product is used. If an element is None, the
                ``concentration`` attribute of the product is used.

        Returns:
            The reaction term to be used in a formulation.
        """

        # make sure products is a list
        products = self.product if isinstance(self.product, list) else [self.product]

        # detect if mixed_domain
        mixed_domain = any(
            isinstance(reactant, _Species) and reactant.subdomain_to_solution != {}
            for reactant in self.reactant
        ) or any(
            isinstance(product, _Species) and product.subdomain_to_solution != {}
            for product in products
        )

        def get_concentration(species):
            if mixed_domain:
                return species.concentration_submesh(self.volume)
            else:
                return species.concentration

        # reaction rates
        k = self._evaluate_rate(self.forward_rate, temperature)
        p = (
            self._evaluate_rate(self.backward_rate, temperature)
            if self.backward_rate is not None
            else 0
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

        return k * product_of_reactants - (p * product_of_products)

    def source_contributions(self, temperature):
        """Express the reaction as a set of volumetric source contributions, one
        per participating :class:`~festim.species.Species`.

        Each contribution is returned in the same sign convention as a
        :class:`~festim.source.SourceBase`, i.e. the returned ``value`` is meant
        to enter the residual as ``-value * test_function * dx``. Reactants are
        consumed (``value = -R``) and products are produced (``value = +R``),
        where ``R`` is :meth:`reaction_term`. Implicit species (which have no
        governing equation) are skipped.

        Arguments:
            temperature: The temperature at which the reaction term is computed.

        Returns:
            A list of ``(species, value)`` tuples.
        """
        rate = self.reaction_term(temperature)

        products = self.product if isinstance(self.product, list) else [self.product]

        contributions = [
            (reactant, -rate)
            for reactant in self.reactant
            if isinstance(reactant, _Species)
        ]
        contributions += [
            (product, rate) for product in products if isinstance(product, _Species)
        ]
        return contributions


class Reaction(GenericReaction):
    """A reaction between two species, with a forward and backward rate built
    from Arrhenius laws. This is typically used to model trapping/detrapping.

    Arguments:
        reactant (Union[F.Species, F.ImplicitSpecies], List[Union[F.Species,
        F.ImplicitSpecies]]): The reactant.
        product (Optional[Union[F.Species, List[F.Species]]]): The product.
        k_0 (float): The forward rate constant pre-exponential factor.
        E_k (float): The forward rate constant activation energy.
        p_0 (float): The backward rate constant pre-exponential factor.
        E_p (float): The backward rate constant activation energy.
        volume (F.VolumeSubdomain1D): The volume subdomain where the reaction
        takes place.

    Attributes:
        reactant (Union[F.Species, F.ImplicitSpecies], List[Union[F.Species,
        F.ImplicitSpecies]]): The reactant.
        product (Optional[Union[F.Species, List[F.Species]]]): The product.
        k_0 (float): The forward rate constant pre-exponential factor.
        E_k (float): The forward rate constant activation energy.
        p_0 (float): The backward rate constant pre-exponential factor.
        E_p (float): The backward rate constant activation energy.
        volume (F.VolumeSubdomain1D): The volume subdomain where the reaction
        takes place.

    Examples:

        :: testsetup:: Reaction

            from festim import Reaction, Species, ImplicitSpecies

        :: testcode:: Reaction

            # create a volume subdomain
            # create two species
            reactant = [F.Species("A"), F.Species("B")]

            # create a product species
            product = F.Species("C")

            # create a reaction between the two species
            reaction = Reaction(reactant, product, k_0=1.0, E_k=0.2, p_0=0.1, E_p=0.3)
            print(reaction)
            # A + B <--> C

            # compute the reaction term at a given temperature
            temperature = 300.0
            reaction_term = reaction.reaction_term(temperature)
    """

    def __init__(
        self,
        reactant: _Species | _ImplicitSpecies | list[_Species | _ImplicitSpecies],
        k_0: float,
        E_k: float,
        volume: VS1D,
        product: Union[_Species, list[_Species]] | None = None,
        p_0: float | None = None,
        E_p: float | None = None,
    ) -> None:
        if product is None:
            product = []

        self.k_0 = k_0
        self.E_k = E_k
        self.p_0 = p_0
        self.E_p = E_p

        def forward_rate(temperature):
            return self.k_0 * exp(-self.E_k / (_k_B * temperature))

        backward_rate = None
        if self.p_0 is not None:

            def backward_rate(temperature):
                if self.p_0 and self.E_p:
                    return self.p_0 * exp(-self.E_p / (_k_B * temperature))
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
        reactant_concentrations: list | None = None,
        product_concentrations: list | None = None,
    ) -> Expr:
        # validate that p_0/E_p are consistent with the presence of products
        # only at this point (not at __init__) so that Reaction objects can be
        # built ahead of time even if p_0/E_p aren't set yet
        if self.product == []:
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
            temperature, reactant_concentrations, product_concentrations
        )

    def __repr__(self) -> str:
        reactants = " + ".join([str(reactant) for reactant in self.reactant])
        if isinstance(self.product, list):
            products = " + ".join([str(product) for product in self.product])
        else:
            products = self.product
        return f"{type(self).__name__}({reactants} <--> {products}, {self.k_0}, {self.E_k}, {self.p_0}, {self.E_p})"  # noqa: E501
