from typing import List


class Species:
    """
    Hydrogen species class for H transport simulation.

    Args:
        name (str, optional): a name given to the species. Defaults to None.

    Attributes:
        name (str): a name given to the species.
        solution (dolfinx.fem.Function or ...): the solution for the current timestep
        prev_solution (dolfinx.fem.Function or ...): the solution for the previous timestep
        test_function (ufl.Argument or ...): the testfunction associated with this species
        concentration (dolfinx.fem.Function): the concentration of the species

    Usage:
        >>> from festim import Species, HTransportProblem
        >>> species = Species(name="H")
        >>> species.name
        'H'
        >>> my_model = HTransportProblem()
        >>> my_model.species.append(species)

    """

    def __init__(self, name: str = None) -> None:
        self.name = name

        self.solution = None
        self.prev_solution = None
        self.test_function = None

    def __repr__(self) -> str:
        return f"Species({self.name})"

    def __str__(self) -> str:
        return f"{self.name}"

    @property
    def concentration(self):
        return self.solution


class Trap(Species):
    """Trap species class for H transport simulation.

    Args:
        name (str, optional): a name given to the trap. Defaults to None.

    Attributes:
        name (str): a name given to the trap.
        attributes of Species class

    Usage:
        >>> from festim import Trap, HTransportProblem
        >>> trap = Trap(name="Trap")
        >>> trap.name
        'Trap'
        >>> my_model = HTransportProblem()
        >>> my_model.species.append(trap)

    """

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


class ImplicitSpecies:
    """Implicit species class for H transport simulation.
    c = n - others

    Args:
        n (float): the total concentration of the species
        others (List[Species]): the list of species from which the implicit
            species concentration is computed (c = n - others)
        name (str, optional): a name given to the species. Defaults to None.

    Attributes:
        name (str): a name given to the species.
        n (float): the total concentration of the species
        others (List[Species]): the list of species from which the implicit
            species concentration is computed (c = n - others)
        concentration (form): the concentration of the species

    """

    def __init__(
        self,
        n: float,
        others: List[Species] = None,
        name: str = None,
    ) -> None:
        self.name = name
        self.n = n
        self.others = others

    def __repr__(self) -> str:
        return f"ImplicitSpecies({self.name}, {self.n}, {self.others})"

    def __str__(self) -> str:
        return f"{self.name}"

    @property
    def concentration(self):
        if len(self.others) > 0:
            for other in self.others:
                if other.solution is None:
                    raise ValueError(
                        f"Cannot compute concentration of {self.name} because {other.name} has no solution"
                    )
        return self.n - sum([other.solution for other in self.others])
