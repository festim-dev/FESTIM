import festim as F
from typing import Union

from ufl import exp


class Reaction:
    """A reaction between two species, with a forward and backward rate.

    Arguments:
        reactant1 (Union[F.Species, F.ImplicitSpecies]): The first reactant.
        reactant2 (Union[F.Species, F.ImplicitSpecies]): The second reactant.
        product (F.Species): The product.
        k_0 (float): The forward rate constant pre-exponential factor.
        E_k (float): The forward rate constant activation energy.
        p_0 (float): The backward rate constant pre-exponential factor.
        E_p (float): The backward rate constant activation energy.

    Attributes:
        reactant1 (Union[F.Species, F.ImplicitSpecies]): The first reactant.
        reactant2 (Union[F.Species, F.ImplicitSpecies]): The second reactant.
        product (F.Species): The product.
        k_0 (float): The forward rate constant pre-exponential factor.
        E_k (float): The forward rate constant activation energy.
        p_0 (float): The backward rate constant pre-exponential factor.
        E_p (float): The backward rate constant activation energy.

    Usage:
        >>> # create two species
        >>> species1 = F.Species("A")
        >>> species2 = F.Species("B")

        >>> # create a product species
        >>> product = F.Species("C")

        >>> # create a reaction between the two species
        >>> reaction = Reaction(species1, species2, product, k_0=1.0, E_k=0.2, p_0=0.1, E_p=0.3)
        >>> print(reaction)
        A + B <--> C

        >>> # compute the reaction term at a given temperature
        >>> temperature = 300.0
        >>> reaction_term = reaction.reaction_term(temperature)

    """

    def __init__(
        self,
        reactant1: Union[F.Species, F.ImplicitSpecies],
        reactant2: Union[F.Species, F.ImplicitSpecies],
        product: F.Species,
        k_0: float,
        E_k: float,
        p_0: float,
        E_p: float,
    ) -> None:
        self.reactant1 = reactant1
        self.reactant2 = reactant2
        self.product = product
        self.k_0 = k_0
        self.E_k = E_k
        self.p_0 = p_0
        self.E_p = E_p

    def __repr__(self) -> str:
        return f"Reaction({self.reactant1} + {self.reactant2} <--> {self.product}, {self.k_0}, {self.E_k}, {self.p_0}, {self.E_p})"

    def __str__(self) -> str:
        return f"{self.reactant1} + {self.reactant2} <--> {self.product}"

    def reaction_term(self, temperature):
        """Compute the reaction term at a given temperature.

        Arguments:
            temperature (): The temperature at which the reaction term is computed.
        """
        k = self.k_0 * exp(-self.E_k / (F.k_B * temperature))
        p = self.p_0 * exp(-self.E_p / (F.k_B * temperature))

        c_A = self.reactant1.concentration
        c_B = self.reactant2.concentration

        c_C = self.product.solution
        return k * c_A * c_B - p * c_C
