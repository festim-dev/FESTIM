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
