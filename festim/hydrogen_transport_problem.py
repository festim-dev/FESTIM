from dolfinx import fem
import ufl

from dolfinx.fem import Function
from ufl import TestFunction


class HydrogenTransportProblem:
    """
    Hydrogen Transport Problem.

    Args:
        mesh (festim.Mesh): the mesh of the model
        subdomains (list of festim.Subdomain): the subdomains of the model
        species (list of festim.Species): the species of the model

    Attributes:
        mesh (festim.Mesh): the mesh of the model
        subdomains (list of festim.Subdomain): the subdomains of the model
        species (list of festim.Species): the species of the model
        temperature (dolfinx.Function): the temperature of the model
        boundary_conditions (list of festim.BoundaryCondition): the boundary conditions of the model
        solver_parameters (dict): the solver parameters of the model
        exports (list of festim.Export): the exports of the model
        dx (dolfinx.fem.dx): the volume measure of the model
        ds (dolfinx.fem.ds): the surface measure of the model
        function_space (dolfinx.fem.FunctionSpace): the function space of the model
        facet_tags (dolfinx.cpp.mesh.MeshTags): the facet tags of the model
        volume_tags (dolfinx.cpp.mesh.MeshTags): the volume tags of the model


    Usage:
        >>> import festim as F
        >>> my_model = F.HydrogenTransportProblem()
        >>> my_model.mesh = F.Mesh(...)
        >>> my_model.subdomains = [F.Subdomain(...)]
        >>> my_model.species = [F.Species(name="H"), F.Species(name="Trap")]
        >>> my_model.initialise()

        or

        >>> my_model = F.HydrogenTransportProblem(
        ...     mesh=F.Mesh(...),
        ...     subdomains=[F.Subdomain(...)],
        ...     species=[F.Species(name="H"), F.Species(name="Trap")],
        ... )
        >>> my_model.initialise()

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
        spaces, facet and volume tags...
        """

        self.define_function_space()
        (
            self.facet_tags,
            self.volume_tags,
            self.dx,
            self.ds,
        ) = self.mesh.create_measures_and_tags(self.function_space)
        self.assign_functions_to_species()

    def define_function_space(self):
        elements = ufl.FiniteElement("CG", self.mesh.mesh.ufl_cell(), 1)
        self.function_space = fem.FunctionSpace(self.mesh.mesh, elements)

    def assign_functions_to_species(self):
        """Creates for each species the solution, prev solution and test function"""
        for spe in self.species:
            spe.solution = Function(self.function_space)
            spe.prev_solution = Function(self.function_space)
            spe.test_function = TestFunction(self.function_space)
