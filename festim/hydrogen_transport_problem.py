import numpy as np
from dolfinx import fem
import ufl


class HydrogenTransportProblem:
    """
    Hydrogen Transport Problem.

    Args:
        mesh (festim.Mesh): the mesh of the model
        subdomains (list of festim.Subdomain):

    Attributes:
        geometry (festim.Geometry): the geometry of the model (mesh, function
            spaces and subdomains)

    """

    def __init__(
        self,
        mesh=None,
        subdomains=[],
        species=[],
        temperature=None,
        boundary_conditions=[],
        solver_parameters=None,
        exports=[],
    ) -> None:
        self.mesh = mesh
        self.subdomains = subdomains
        self.species = species
        self.temperature = temperature
        self.boundary_conditions = boundary_conditions
        self.solver_parameters = solver_parameters
        self.exports = exports

        self.dx = None
        self.ds = None
        self.function_space = None
        self.facet_tags = None
        self.volume_tags = None

    def initialise(self):
        """Initialise the model. Creates suitable function
        spaces, the subdomains...
        """

        self.define_function_space()
        (
            self.dx,
            self.ds,
            self.facet_tags,
            self.volume_tags,
        ) = self.mesh.create_measures_and_tags(self.function_space)

    def define_function_space(self):
        elements = ufl.FiniteElement("CG", self.mesh.mesh.ufl_cell(), 1)
        self.function_space = fem.FunctionSpace(self.mesh.mesh, elements)
