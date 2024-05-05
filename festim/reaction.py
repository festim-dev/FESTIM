import festim as F
from typing import Union, Optional, List

from ufl import exp


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

    Usage:
        >>> # create two species
        >>> reactant = [F.Species("A"), F.Species("B")]

        >>> # create a product species
        >>> product = F.Species("C")

        >>> # create a reaction between the two species
        >>> reaction = Reaction(reactant, product, k_0=1.0, E_k=0.2, p_0=0.1, E_p=0.3)
        >>> print(reaction)
        A + B <--> C

        >>> # compute the reaction term at a given temperature
        >>> temperature = 300.0
        >>> reaction_term = reaction.reaction_term(temperature)

    """

    def __init__(
        self,
        reactant: Union[
            F.Species, F.ImplicitSpecies, List[Union[F.Species, F.ImplicitSpecies]]
        ],
        k_0: float,
        E_k: float,
        volume: F.VolumeSubdomain1D,
        product: Optional[Union[F.Species, List[F.Species]]] = [],
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
                f"reactant must be an entry of one or more species objects, not an empty list."
            )
        for i in value:
            if not isinstance(i, (F.Species, F.ImplicitSpecies)):
                raise TypeError(
                    f"reactant must be an F.Species or F.ImplicitSpecies, not {type(i)}"
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

    def reaction_term(self, temperature):
        """Compute the reaction term at a given temperature.

        Arguments:
            temperature (): The temperature at which the reaction term is computed.
        """

        if self.product == []:
            if self.p_0 is not None:
                raise ValueError(
                    f"p_0 must be None, not {self.p_0} when no products are present."
                )
            if self.E_p is not None:
                raise ValueError(
                    f"E_p must be None, not {self.E_p} when no products are present."
                )

        k = self.k_0 * exp(-self.E_k / (F.k_B * temperature))

        if self.p_0 and self.E_p:
            p = self.p_0 * exp(-self.E_p / (F.k_B * temperature))
        elif self.p_0:
            p = self.p_0
        else:
            p = 0

        reactants = self.reactant
        product_of_reactants = reactants[0].concentration
        for reactant in reactants[1:]:
            product_of_reactants *= reactant.concentration

        if isinstance(self.product, list):
            products = self.product
        else:
            products = [self.product]

        if len(products) > 0:
            product_of_products = products[0].solution
            for product in products[1:]:
                product_of_products *= product.solution
        else:
            product_of_products = 0
        return k * product_of_reactants - (p * product_of_products)
