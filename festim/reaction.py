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
        volume (F.VolumeSubdomain1D): The volume subdomain where the reaction takes place.

    Attributes:
        reactant1 (Union[F.Species, F.ImplicitSpecies]): The first reactant.
        reactant2 (Union[F.Species, F.ImplicitSpecies]): The second reactant.
        product (F.Species): The product.
        k_0 (float): The forward rate constant pre-exponential factor.
        E_k (float): The forward rate constant activation energy.
        p_0 (float): The backward rate constant pre-exponential factor.
        E_p (float): The backward rate constant activation energy.
        volume (F.VolumeSubdomain1D): The volume subdomain where the reaction takes place.

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
        volume: F.VolumeSubdomain1D,
    ) -> None:
        self.reactant1 = reactant1
        self.reactant2 = reactant2
        self.product = product
        self.k_0 = k_0
        self.E_k = E_k
        self.p_0 = p_0
        self.E_p = E_p
        self.volume = volume

    @property
    def reactant1(self):
        return self._reactant1

    @reactant1.setter
    def reactant1(self, value):
        if not isinstance(value, (F.Species, F.ImplicitSpecies)):
            raise TypeError(
                f"reactant1 must be an F.Species or F.ImplicitSpecies, not {type(value)}"
            )
        self._reactant1 = value

    @property
    def reactant2(self):
        return self._reactant2

    @reactant2.setter
    def reactant2(self, value):
        if not isinstance(value, (F.Species, F.ImplicitSpecies)):
            raise TypeError(
                f"reactant2 must be an F.Species or F.ImplicitSpecies, not {type(value)}"
            )
        self._reactant2 = value

    def __repr__(self) -> str:
        if isinstance(self.product, list):
            products = " + ".join([str(product) for product in self.product])
        else:
            products = self.product
        return f"Reaction({self.reactant1} + {self.reactant2} <--> {products}, {self.k_0}, {self.E_k}, {self.p_0}, {self.E_p})"

    def __str__(self) -> str:
        if isinstance(self.product, list):
            products = " + ".join([str(product) for product in self.product])
        else:
            products = self.product
        return f"{self.reactant1} + {self.reactant2} <--> {products}"

    def reaction_term(self, temperature):
        """Compute the reaction term at a given temperature.

        Arguments:
            temperature (): The temperature at which the reaction term is computed.
        """
        k = self.k_0 * exp(-self.E_k / (F.k_B * temperature))
        p = self.p_0 * exp(-self.E_p / (F.k_B * temperature))

        c_A = self.reactant1.concentration
        c_B = self.reactant2.concentration

        if isinstance(self.product, list):
            products = self.product
        else:
            products = [self.product]

        products_of_product = products[0].solution
        for product in products[1:]:
            products_of_product *= product.solution

        return k * c_A * c_B - p * products_of_product
