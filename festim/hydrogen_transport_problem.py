import numpy as np


class HydrogenTransportProblem:
    """
    Hydrogen Transport Problem.
    Used internally in festim.Model

    Args:
        geometry (festim.Geometry): the geometry of the model (mesh, function
            spaces and subdomains)

    Attributes:
        geometry (festim.Geometry): the geometry of the model (mesh, function
            spaces and subdomains)

    """

    def __init__(
        self,
        geometry=None,
        species=[],
        temperature=None,
        boundary_conditions=[],
        solver_parameters=None,
        exports=[],
    ) -> None:
        self.geometry = geometry
        self.species = species
        self.temperature = temperature
        self.boundary_conditions = boundary_conditions
        self.solver_parameters = solver_parameters
        self.exports = exports

        self.dx = None
        self.ds = None
        self.function_spaces = []

    def initialise(self):
        """Initialise the model. Creates suitable function
        spaces, the subdomains...
        """

        self.function_spaces.append(self.geometry.define_function_space())
        self.geometry.define_subdomains(self.function_spaces[0])
