from typing import Optional, Union, List

from ufl import exp
from ufl.core.expr import Expr

from festim import k_B as _k_B
from festim.species import ImplicitSpecies as _ImplicitSpecies
from festim.species import Species as _Species
from festim.subdomain.volume_subdomain_1d import VolumeSubdomain1D as VS1D


class Reaction:
    """A reaction between two species, with a forward and backward rate.

    Arguments:
        reactant (Union[F.Species, F.ImplicitSpecies], List[Union[F.Species, F.ImplicitSpecies]]): The reactant.
        product (Optional[Union[F.Species, List[F.Species]]]): The product.
        k_0 (float): The forward rate constant pre-exponential factor.
        E_k (float): The forward rate constant activation energy.
        p_0 (float): The backward rate constant pre-exponential factor.
        E_p (float): The backward rate constant activation energy.
        volume (F.VolumeSubdomain1D): The volume subdomain where the reaction takes place.

    Attributes:
        reactant (Union[F.Species, F.ImplicitSpecies], List[Union[F.Species, F.ImplicitSpecies]]): The reactant.
        product (Optional[Union[F.Species, List[F.Species]]]): The product.
        k_0 (float): The forward rate constant pre-exponential factor.
        E_k (float): The forward rate constant activation energy.
        p_0 (float): The backward rate constant pre-exponential factor.
        E_p (float): The backward rate constant activation energy.
        volume (F.VolumeSubdomain1D): The volume subdomain where the reaction takes place.

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
        product: Optional[Union[_Species, list[_Species]]] = [],
        p_0: float = None,
        E_p: float = None,
    ) -> None:
        self.k_0 = k_0
        self.E_k = E_k
        self.p_0 = p_0
        self.E_p = E_p
        self.reactant = reactant
        self.product = product
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
                "reactant must be an entry of one or more species objects, not an empty list."
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
        return f"Reaction({reactants} <--> {products}, {self.k_0}, {self.E_k}, {self.p_0}, {self.E_p})"

    def __str__(self) -> str:
        reactants = " + ".join([str(reactant) for reactant in self.reactant])
        if isinstance(self.product, list):
            products = " + ".join([str(product) for product in self.product])
        else:
            products = self.product
        return f"{reactants} <--> {products}"

    def reaction_term(
        self,
        temperature,
        reactant_concentrations: List = None,
        product_concentrations: List = None,
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
            if self.p_0 == None:
                raise ValueError(
                    "p_0 cannot be None when reaction products are present."
                )
            elif self.E_p == None:
                raise ValueError(
                    "E_p cannot be None when reaction products are present."
                )

        products = self.product if isinstance(self.product, list) else [self.product]

        # reaction rates
        k = self.k_0 * exp(-self.E_k / (_k_B * temperature))

        if self.p_0 and self.E_p:
            p = self.p_0 * exp(-self.E_p / (_k_B * temperature))
        elif self.p_0:
            p = self.p_0
        else:
            p = 0

        # if reactant_concentrations is provided, use these concentrations
        reactants = self.reactant
        if reactant_concentrations is not None:
            assert len(reactant_concentrations) == len(reactants)
            for i, reactant in enumerate(reactants):
                if reactant_concentrations[i] is None:
                    reactant_concentrations[i] = reactant.concentration
        else:
            reactant_concentrations = [reactant.concentration for reactant in reactants]

        # if product_concentrations is provided, use these concentrations
        if product_concentrations is not None:
            assert len(product_concentrations) == len(products)
            for i, product in enumerate(products):
                if product_concentrations[i] is None:
                    product_concentrations[i] = product.concentration
        else:
            product_concentrations = [product.concentration for product in products]

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
