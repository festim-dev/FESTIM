from dolfinx import fem
from dolfinx.nls.petsc import NewtonSolver
import ufl
from mpi4py import MPI
from dolfinx.fem import Function
from ufl import (
    TestFunction,
    dot,
    grad,
    exp,
)

import festim as F

class HydrogenTransportProblem:
    """
    Hydrogen Transport Problem.

    Args:
        mesh (festim.Mesh): the mesh of the model
        subdomains (list of festim.Subdomain): the subdomains of the model
        species (list of festim.Species): the species of the model
        temperature (float or dolfinx.Function): the temperature of the model
        sources (list of festim.Source): the hydrogen sources of the model
        boundary_conditions (list of festim.BoundaryCondition): the boundary conditions of the model
        solver_parameters (dict): the solver parameters of the model
        exports (list of festim.Export): the exports of the model

    Attributes:
        mesh (festim.Mesh): the mesh of the model
        subdomains (list of festim.Subdomain): the subdomains of the model
        species (list of festim.Species): the species of the model
        temperature (float or dolfinx.Function): the temperature of the model
        boundary_conditions (list of festim.BoundaryCondition): the boundary conditions of the model
        solver_parameters (dict): the solver parameters of the model
        exports (list of festim.Export): the exports of the model
        dx (dolfinx.fem.dx): the volume measure of the model
        ds (dolfinx.fem.ds): the surface measure of the model
        function_space (dolfinx.fem.FunctionSpace): the function space of the model
        facet_tags (dolfinx.cpp.mesh.MeshTags): the facet tags of the model
        volume_tags (dolfinx.cpp.mesh.MeshTags): the volume tags of the model
        formulation (ufl.form.Form): the formulation of the model


    Usage:
        >>> import festim as F
        >>> my_model = F.HydrogenTransportProblem()
        >>> my_model.mesh = F.Mesh(...)
        >>> my_model.subdomains = [F.Subdomain(...)]
        >>> my_model.species = [F.Species(name="H"), F.Species(name="Trap")]
        >>> my_model.temperature = 500
        >>> my_model.sources = [F.Source(...)]
        >>> my_model.boundary_conditions = [F.BoundaryCondition(...)]
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
        sources=[],
        boundary_conditions=[],
        solver_parameters=None,
        exports=[],
    ) -> None:
        self.mesh = mesh
        self.subdomains = subdomains
        self.species = species
        self.temperature = temperature
        self.sources = sources
        self.boundary_conditions = boundary_conditions
        self.solver_parameters = solver_parameters
        self.exports = exports

        self.dx = None
        self.ds = None
        self.function_space = None
        self.facet_tags = None
        self.volume_tags = None
        self.formulation = None

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
        self.create_formulation()
        self.create_solver()

    def define_function_space(self):
        elements = ufl.FiniteElement("CG", self.mesh.mesh.ufl_cell(), 1)
        self.function_space = fem.FunctionSpace(self.mesh.mesh, elements)

    def assign_functions_to_species(self):
        """Creates for each species the solution, prev solution and test function
        """
        if len(self.species) > 1:
            raise NotImplementedError("Multiple species not implemented yet")
        for spe in self.species:
            spe.solution = Function(self.function_space)
            spe.prev_solution = Function(self.function_space)
            spe.test_function = TestFunction(self.function_space)

    def create_formulation(self):
        """Creates the formulation of the model"""
        # f = Constant(my_mesh.mesh, (PETSc.ScalarType(0)))
        if len(self.sources) > 1:
            raise NotImplementedError("Sources not implemented yet")
        if len(self.subdomains) > 1:
            raise NotImplementedError("Multiple subdomains not implemented yet")
        if len(self.species) > 1:
            raise NotImplementedError("Multiple species not implemented yet")

        # TODO expose D_0 and E_D as parameters of a Material class
        D_0 = fem.Constant(self.mesh.mesh, 1.9e-7)
        E_D = fem.Constant(self.mesh.mesh, 0.2)


        D = D_0 * exp(-E_D / F.k_B / self.temperature)

        dt = fem.Constant(self.mesh.mesh, 1 / 20)

        self.D = D # TODO remove this
        self.dt = dt # TODO remove this

        u = self.species[0].solution
        u_n = self.species[0].prev_solution
        v = self.species[0].test_function
        formulation = dot(D * grad(u), grad(v)) * self.dx
        formulation += ((u - u_n) / dt) * v * self.dx

        self.formulation = formulation
    
    def create_solver(self):
        """Creates the solver of the model"""
        problem = fem.petsc.NonlinearProblem(self.formulation, self.species[0].solution, bcs=self.boundary_conditions)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        self.solver = solver
