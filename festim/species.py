class Species:
    """
    Hydrogen Transport Problem.

    Args:
        name (str, optional): a name given to the species. Defaults to None.

    Attributes:
        name (str): a name given to the species.
        solution (dolfinx.Function or ...): the solution for the current timestep
        prev_solution (dolfinx.Function or ...): the solution for the previous timestep
        test_function (ufl.TestFunction or ...): the testfunction associated with this species
    """
    def __init__(self, name:str=None) -> None:
        """_summary_

        Args:
            name (str, optional): a name given to the species. Defaults to None.
        """
        self.name = name

        self.solution = None
        self.prev_solution = None
        self.test_function = None

    
class Trap(Species):
    def __init__(self, name:str=None) -> None:
        super().__init__(name)