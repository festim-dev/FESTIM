import dolfinx
from dolfinx import fem
from dolfinx.nls.petsc import NewtonSolver
import basix
import basix.ufl
import ufl
from mpi4py import MPI
import numpy as np
import tqdm.autonotebook
import festim as F

from dolfinx.mesh import meshtags


class HydrogenTransportProblem:
    """
    Hydrogen Transport Problem.

    Args:
        mesh (festim.Mesh): the mesh of the model
        subdomains (list of festim.Subdomain): the subdomains of the model
        species (list of festim.Species): the species of the model
        reactions (list of festim.Reaction): the reactions of the model
        temperature (float, int, fem.Constant, fem.Function or callable): the
            temperature of the model (K)
        sources (list of festim.Source): the hydrogen sources of the model
        initial_conditions (list of festim.InitialCondition): the initial conditions
            of the model
        boundary_conditions (list of festim.BoundaryCondition): the boundary
            conditions of the model
        solver_parameters (dict): the solver parameters of the model
        exports (list of festim.Export): the exports of the model
        traps (list of F.Trap): the traps of the model

    Attributes:
        mesh (festim.Mesh): the mesh of the model
        subdomains (list of festim.Subdomain): the subdomains of the model
        species (list of festim.Species): the species of the model
        reactions (list of festim.Reaction): the reactions of the model
        temperature (float, int, fem.Constant, fem.Function or callable): the
            temperature of the model (K)
        sources (list of festim.Source): the hydrogen sources of the model
        initial_conditions (list of festim.InitialCondition): the initial conditions
            of the model
        boundary_conditions (list of festim.BoundaryCondition): the boundary
            conditions of the model
        solver_parameters (dict): the solver parameters of the model
        exports (list of festim.Export): the exports of the model
        traps (list of F.Trap): the traps of the model
        dx (dolfinx.fem.dx): the volume measure of the model
        ds (dolfinx.fem.ds): the surface measure of the model
        function_space (dolfinx.fem.FunctionSpaceBase): the function space of the
            model
        facet_meshtags (dolfinx.mesh.MeshTags): the facet meshtags of the model
        volume_meshtags (dolfinx.mesh.MeshTags): the volume meshtags of the
            model
        formulation (ufl.form.Form): the formulation of the model
        solver (dolfinx.nls.newton.NewtonSolver): the solver of the model
        multispecies (bool): True if the model has more than one species.
        temperature_fenics (fem.Constant or fem.Function): the
            temperature of the model as a fenics object (fem.Constant or
            fem.Function).
        temperature_expr (fem.Expression): the expression of the temperature
            that is used to update the temperature_fenics
        temperature_time_dependent (bool): True if the temperature is time
            dependent
        V_DG_0 (dolfinx.fem.FunctionSpaceBase): A DG function space of degree 0
            over domain
        V_DG_1 (dolfinx.fem.FunctionSpaceBase): A DG function space of degree 1
            over domain
        volume_subdomains (list of festim.VolumeSubdomain): the volume subdomains
            of the model
        surface_subdomains (list of festim.SurfaceSubdomain): the surface subdomains
            of the model


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
        subdomains=None,
        species=None,
        reactions=None,
        temperature=None,
        sources=None,
        initial_conditions=None,
        boundary_conditions=None,
        settings=None,
        exports=None,
        traps=None,
    ):
        self.mesh = mesh
        self.temperature = temperature
        self.settings = settings

        # for arguments to initliase as empty list
        # if arg not None, assign arg, else assign empty list
        self.subdomains = subdomains or []
        self.species = species or []
        self.reactions = reactions or []
        self.sources = sources or []
        self.initial_conditions = initial_conditions or []
        self.boundary_conditions = boundary_conditions or []
        self.exports = exports or []
        self.traps = traps or []

        self.dx = None
        self.ds = None
        self.function_space = None
        self.facet_meshtags = None
        self.volume_meshtags = None
        self.formulation = None
        self.bc_forms = []
        self.temperature_fenics = None

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value is None:
            self._temperature = value
        elif isinstance(value, (float, int, fem.Constant, fem.Function)):
            self._temperature = value
        elif callable(value):
            self._temperature = value
        else:
            raise TypeError(
                f"Value must be a float, int, fem.Constant, fem.Function, or callable"
            )

    @property
    def temperature_fenics(self):
        return self._temperature_fenics

    @temperature_fenics.setter
    def temperature_fenics(self, value):
        if value is None:
            self._temperature_fenics = value
            return
        elif not isinstance(
            value,
            (fem.Constant, fem.Function),
        ):
            raise TypeError(f"Value must be a fem.Constant or fem.Function")
        self._temperature_fenics = value

    @property
    def temperature_time_dependent(self):
        if self.temperature is None:
            return False
        if isinstance(self.temperature, fem.Constant):
            return False
        if callable(self.temperature):
            arguments = self.temperature.__code__.co_varnames
            return "t" in arguments
        else:
            return False

    @property
    def multispecies(self):
        return len(self.species) > 1

    @property
    def volume_subdomains(self):
        return [s for s in self.subdomains if isinstance(s, F.VolumeSubdomain)]

    @property
    def surface_subdomains(self):
        return [s for s in self.subdomains if isinstance(s, F.SurfaceSubdomain)]

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        # check that all species are of type festim.Species
        for spe in value:
            if not isinstance(spe, F.Species):
                raise TypeError(
                    f"elements of species must be of type festim.Species not {type(spe)}"
                )
        self._species = value

    @property
    def facet_meshtags(self):
        return self._facet_meshtags

    @facet_meshtags.setter
    def facet_meshtags(self, value):
        if value is None:
            self._facet_meshtags = value
        elif isinstance(value, dolfinx.mesh.MeshTags):
            self._facet_meshtags = value
        else:
            raise TypeError(f"value must be of type dolfinx.mesh.MeshTags")

    @property
    def volume_meshtags(self):
        return self._volume_meshtags

    @volume_meshtags.setter
    def volume_meshtags(self, value):
        if value is None:
            self._volume_meshtags = value
        elif isinstance(value, dolfinx.mesh.MeshTags):
            self._volume_meshtags = value
        else:
            raise TypeError(f"value must be of type dolfinx.mesh.MeshTags")

    def initialise(self):
        self.create_species_from_traps()
        self.define_function_spaces()
        self.define_meshtags_and_measures()
        self.assign_functions_to_species()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self.dt = F.as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )

        self.define_temperature()
        self.define_boundary_conditions()
        self.create_source_values_fenics()
        self.create_initial_conditions()
        self.create_formulation()
        self.create_solver()
        self.initialise_exports()

    def create_species_from_traps(self):
        """Generate a species and reaction per trap defined in self.traps"""

        for trap in self.traps:
            trap.create_species_and_reaction()
            self.species.append(trap.trapped_concentration)
            self.reactions.append(trap.reaction)

    def define_temperature(self):
        """Sets the value of temperature_fenics_value. The type depends on
        self.temperature. If self.temperature is a function on t only, create
        a fem.Constant. Else, create an dolfinx.fem.Expression (stored in
        self.temperature_expr) to be updated, a dolfinx.fem.Function object
        is created from the Expression (stored in self.temperature_fenics_value).
        Raise a ValueError if temperature is None.
        """
        # check if temperature is None
        if self.temperature is None:
            raise ValueError("the temperature attribute needs to be defined")

        # if temperature is a float or int, create a fem.Constant
        elif isinstance(self.temperature, (float, int)):
            self.temperature_fenics = F.as_fenics_constant(
                self.temperature, self.mesh.mesh
            )
        # if temperature is a fem.Constant or function, pass it to temperature_fenics
        elif isinstance(self.temperature, (fem.Constant, fem.Function)):
            self.temperature_fenics = self.temperature

        # if temperature is callable, process accordingly
        elif callable(self.temperature):
            arguments = self.temperature.__code__.co_varnames
            if "t" in arguments and "x" not in arguments:
                if not isinstance(self.temperature(t=float(self.t)), (float, int)):
                    raise ValueError(
                        f"self.temperature should return a float or an int, not {type(self.temperature(t=float(self.t)))} "
                    )
                # only t is an argument
                self.temperature_fenics = F.as_fenics_constant(
                    mesh=self.mesh.mesh, value=self.temperature(t=float(self.t))
                )
            else:
                x = ufl.SpatialCoordinate(self.mesh.mesh)
                degree = 1
                element_temperature = basix.ufl.element(
                    basix.ElementFamily.P,
                    self.mesh.mesh.basix_cell(),
                    degree,
                    basix.LagrangeVariant.equispaced,
                )
                function_space_temperature = fem.functionspace(
                    self.mesh.mesh, element_temperature
                )
                self.temperature_fenics = fem.Function(function_space_temperature)
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = self.t
                if "x" in arguments:
                    kwargs["x"] = x

                # store the expression of the temperature
                # to update the temperature_fenics later
                self.temperature_expr = fem.Expression(
                    self.temperature(**kwargs),
                    function_space_temperature.element.interpolation_points(),
                )
                self.temperature_fenics.interpolate(self.temperature_expr)

    def initialise_exports(self):
        """Defines the export writers of the model, if field is given as
        a string, find species object in self.species"""

        for export in self.exports:
            # if name of species is given then replace with species object
            if isinstance(export.field, list):
                for idx, field in enumerate(export.field):
                    if isinstance(field, str):
                        export.field[idx] = F.find_species_from_name(
                            field, self.species
                        )
            elif isinstance(export.field, str):
                export.field = F.find_species_from_name(export.field, self.species)

            if isinstance(export, (F.VTXExport, F.XDMFExport)):
                export.define_writer(MPI.COMM_WORLD)
                if isinstance(export, F.XDMFExport):
                    export.writer.write_mesh(self.mesh.mesh)

        # compute diffusivity function for surface fluxes

        spe_to_D_global = {}  # links species to global D function
        spe_to_D_global_expr = {}  # links species to D expression

        for export in self.exports:
            if isinstance(export, F.SurfaceQuantity):
                if export.field in spe_to_D_global:
                    # if already computed then use the same D
                    D = spe_to_D_global[export.field]
                    D_expr = spe_to_D_global_expr[export.field]
                else:
                    # compute D and add it to the dict
                    D, D_expr = self.define_D_global(export.field)
                    spe_to_D_global[export.field] = D
                    spe_to_D_global_expr[export.field] = D_expr

                # add the global D to the export
                export.D = D
                export.D_expr = D_expr

    def define_D_global(self, species):
        """Defines the global diffusion coefficient for a given species

        Args:
            species (F.Species): the species

        Returns:
            dolfinx.fem.Function, dolfinx.fem.Expression: the global diffusion
                coefficient and the expression of the global diffusion coefficient
                for a given species
        """
        assert isinstance(species, F.Species)

        D_0 = fem.Function(self.V_DG_0)
        E_D = fem.Function(self.V_DG_0)
        for vol in self.volume_subdomains:
            cell_indices = vol.locate_subdomain_entities(self.mesh.mesh, self.mesh.vdim)

            # replace values of D_0 and E_D by values from the material
            D_0.x.array[cell_indices] = vol.material.get_D_0(species=species)
            E_D.x.array[cell_indices] = vol.material.get_E_D(species=species)

        # create global D function
        D = fem.Function(self.V_DG_1)

        expr = D_0 * ufl.exp(
            -E_D / F.as_fenics_constant(F.k_B, self.mesh.mesh) / self.temperature_fenics
        )
        D_expr = fem.Expression(expr, self.V_DG_1.element.interpolation_points())
        D.interpolate(D_expr)
        return D, D_expr

    def define_function_spaces(self):
        """Creates the function space of the model, creates a mixed element if
        model is multispecies. Creates the main solution and previous solution
        function u and u_n. Create global DG function spaces of degree 0 and 1
        for the global diffusion coefficient"""

        degree = 1
        element_CG = basix.ufl.element(
            basix.ElementFamily.P,
            self.mesh.mesh.basix_cell(),
            degree,
            basix.LagrangeVariant.equispaced,
        )
        if not self.multispecies:
            element = element_CG
        else:
            elements = []
            for spe in self.species:
                if isinstance(spe, F.Species):
                    elements.append(element_CG)
            element = basix.ufl.mixed_element(elements)

        self.function_space = fem.functionspace(self.mesh.mesh, element)

        # create global DG function spaces of degree 0 and 1
        element_DG0 = basix.ufl.element(
            "DG",
            self.mesh.mesh.basix_cell(),
            0,
            basix.LagrangeVariant.equispaced,
        )
        element_DG1 = basix.ufl.element(
            "DG",
            self.mesh.mesh.basix_cell(),
            1,
            basix.LagrangeVariant.equispaced,
        )
        self.V_DG_0 = fem.functionspace(self.mesh.mesh, element_DG0)
        self.V_DG_1 = fem.functionspace(self.mesh.mesh, element_DG1)

        self.u = fem.Function(self.function_space)
        self.u_n = fem.Function(self.function_space)

    def assign_functions_to_species(self):
        """Creates the solution, prev solution, test function and
        post-processing solution for each species, if model is multispecies,
        created a collapsed function space for each species"""

        if not self.multispecies:
            sub_solutions = [self.u]
            sub_prev_solution = [self.u_n]
            sub_test_functions = [ufl.TestFunction(self.function_space)]
            self.species[0].sub_function_space = self.function_space
            self.species[0].post_processing_solution = self.u
        else:
            sub_solutions = list(ufl.split(self.u))
            sub_prev_solution = list(ufl.split(self.u_n))
            sub_test_functions = list(ufl.TestFunctions(self.function_space))

            for idx, spe in enumerate(self.species):
                spe.sub_function_space = self.function_space.sub(idx)
                spe.post_processing_solution = self.u.sub(idx)
                spe.collapsed_function_space, _ = self.function_space.sub(
                    idx
                ).collapse()

        for idx, spe in enumerate(self.species):
            spe.solution = sub_solutions[idx]
            spe.prev_solution = sub_prev_solution[idx]
            spe.test_function = sub_test_functions[idx]

    def define_meshtags_and_measures(self):
        """Defines the facet and volume meshtags of the model which are used
        to define the measures fo the model, dx and ds"""

        if isinstance(self.mesh, F.MeshFromXDMF):
            self.facet_meshtags = self.mesh.define_surface_meshtags()
            self.volume_meshtags = self.mesh.define_volume_meshtags()

        elif isinstance(self.mesh, F.Mesh1D):
            self.facet_meshtags, self.volume_meshtags = self.mesh.define_meshtags(
                surface_subdomains=self.surface_subdomains,
                volume_subdomains=self.volume_subdomains,
            )

        elif isinstance(self.mesh, F.Mesh):
            if not self.facet_meshtags:
                # create empty facet_meshtags
                facet_indices = np.array([], dtype=np.int32)
                facet_tags = np.array([], dtype=np.int32)
                self.facet_meshtags = meshtags(
                    self.mesh.mesh, self.mesh.fdim, facet_indices, facet_tags
                )

            if not self.volume_meshtags:
                # create meshtags with all cells tagged as 1
                num_cells = self.mesh.mesh.topology.index_map(self.mesh.vdim).size_local
                mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
                tags_volumes = np.full(num_cells, 1, dtype=np.int32)
                self.volume_meshtags = meshtags(
                    self.mesh.mesh, self.mesh.vdim, mesh_cell_indices, tags_volumes
                )

        # check volume ids are unique
        vol_ids = [vol.id for vol in self.volume_subdomains]
        if len(vol_ids) != len(np.unique(vol_ids)):
            raise ValueError("Volume ids are not unique")

        # define measures
        self.ds = ufl.Measure(
            "ds", domain=self.mesh.mesh, subdomain_data=self.facet_meshtags
        )
        self.dx = ufl.Measure(
            "dx", domain=self.mesh.mesh, subdomain_data=self.volume_meshtags
        )

    def define_boundary_conditions(self):
        """Defines the dirichlet boundary conditions of the model"""
        for bc in self.boundary_conditions:
            if isinstance(bc.species, str):
                # if name of species is given then replace with species object
                bc.species = F.find_species_from_name(bc.species, self.species)
            if isinstance(bc, F.DirichletBC):
                form = self.create_dirichletbc_form(bc)
                self.bc_forms.append(form)

    def create_dirichletbc_form(self, bc):
        """Creates a dirichlet boundary condition form

        Args:
            bc (festim.DirichletBC): the boundary condition

        Returns:
            dolfinx.fem.bcs.DirichletBC: A representation of
                the boundary condition for modifying linear systems.
        """
        # create value_fenics
        function_space_value = None

        if callable(bc.value):
            # if bc.value is a callable then need to provide a functionspace
            if not self.multispecies:
                function_space_value = bc.species.sub_function_space
            else:
                function_space_value = bc.species.collapsed_function_space

        bc.create_value(
            mesh=self.mesh.mesh,
            temperature=self.temperature_fenics,
            function_space=function_space_value,
            t=self.t,
        )

        # get dofs
        if self.multispecies and isinstance(bc.value_fenics, (fem.Function)):
            function_space_dofs = (
                bc.species.sub_function_space,
                bc.species.collapsed_function_space,
            )
        else:
            function_space_dofs = bc.species.sub_function_space

        bc_dofs = bc.define_surface_subdomain_dofs(
            facet_meshtags=self.facet_meshtags,
            mesh=self.mesh,
            function_space=function_space_dofs,
        )

        # create form
        if not self.multispecies and isinstance(bc.value_fenics, (fem.Function)):
            # no need to pass the functionspace since value_fenics is already a Function
            function_space_form = None
        else:
            function_space_form = bc.species.sub_function_space

        form = fem.dirichletbc(
            value=bc.value_fenics,
            dofs=bc_dofs,
            V=function_space_form,
        )

        return form

    def create_source_values_fenics(self):
        """For each source create the value_fenics"""
        for source in self.sources:
            # create value_fenics for all F.Source objects
            if isinstance(source, F.Source):
                function_space_value = None
                if callable(source.value):
                    # if bc.value is a callable then need to provide a functionspace
                    if not self.multispecies:
                        function_space_value = source.species.sub_function_space
                    else:
                        function_space_value = source.species.collapsed_function_space

                source.create_value_fenics(
                    mesh=self.mesh.mesh,
                    temperature=self.temperature_fenics,
                    function_space=function_space_value,
                    t=self.t,
                )

    def create_initial_conditions(self):
        """For each initial condition, create the value_fenics and assign it to
        the previous solution of the condition's species"""

        if len(self.initial_conditions) > 0 and not self.settings.transient:
            raise ValueError(
                "Initial conditions can only be defined for transient simulations"
            )

        function_space_value = None

        for condition in self.initial_conditions:
            # create value_fenics for condition
            function_space_value = None
            if callable(condition.value):
                # if bc.value is a callable then need to provide a functionspace
                if not self.multispecies:
                    function_space_value = condition.species.sub_function_space
                else:
                    function_space_value = condition.species.collapsed_function_space

            condition.create_expr_fenics(
                mesh=self.mesh.mesh,
                temperature=self.temperature_fenics,
                function_space=function_space_value,
            )

            # assign to previous solution of species
            if not self.multispecies:
                condition.species.prev_solution.interpolate(condition.expr_fenics)
            else:
                idx = self.species.index(condition.species)
                self.u_n.sub(idx).interpolate(condition.expr_fenics)

    def create_formulation(self):
        """Creates the formulation of the model"""

        self.formulation = 0

        # add diffusion and time derivative for each species
        for spe in self.species:
            u = spe.solution
            u_n = spe.prev_solution
            v = spe.test_function

            for vol in self.volume_subdomains:
                D = vol.material.get_diffusion_coefficient(
                    self.mesh.mesh, self.temperature_fenics, spe
                )
                if spe.mobile:
                    self.formulation += ufl.dot(D * ufl.grad(u), ufl.grad(v)) * self.dx(
                        vol.id
                    )

                if self.settings.transient:
                    self.formulation += ((u - u_n) / self.dt) * v * self.dx(vol.id)

        for reaction in self.reactions:
            # reactant 1
            if isinstance(reaction.reactant1, F.Species):
                self.formulation += (
                    reaction.reaction_term(self.temperature_fenics)
                    * reaction.reactant1.test_function
                    * self.dx(reaction.volume.id)
                )
            # reactant 2
            if isinstance(reaction.reactant2, F.Species):
                self.formulation += (
                    reaction.reaction_term(self.temperature_fenics)
                    * reaction.reactant2.test_function
                    * self.dx(reaction.volume.id)
                )
            # product
            if isinstance(reaction.product, list):
                products = reaction.product
            else:
                products = [reaction.product]
            for product in products:
                self.formulation += (
                    -reaction.reaction_term(self.temperature_fenics)
                    * product.test_function
                    * self.dx(reaction.volume.id)
                )
        # add sources
        for source in self.sources:
            self.formulation -= (
                source.value_fenics
                * source.species.test_function
                * self.dx(source.volume.id)
            )

        # add fluxes
        # TODO implement this
        # for bc in self.boundary_conditions:
        #     pass
        #     if bc.species == spe and bc.type != "dirichlet":
        #         formulation += bc * v * self.ds

        # check if each species is defined in all volumes
        if not self.settings.transient:
            for spe in self.species:
                # if species mobile, already defined in diffusion term
                if not spe.mobile:
                    not_defined_in_volume = self.volume_subdomains.copy()
                    for vol in self.volume_subdomains:
                        # check reactions
                        for reaction in self.reactions:
                            if vol == reaction.volume:
                                not_defined_in_volume.remove(vol)

                    # add c = 0 to formulation where needed
                    for vol in not_defined_in_volume:
                        self.formulation += (
                            spe.solution * spe.test_function * self.dx(vol.id)
                        )

    def create_solver(self):
        """Creates the solver of the model"""
        problem = fem.petsc.NonlinearProblem(
            self.formulation,
            self.u,
            bcs=self.bc_forms,
        )
        self.solver = NewtonSolver(MPI.COMM_WORLD, problem)
        self.solver.atol = self.settings.atol
        self.solver.rtol = self.settings.rtol
        self.solver.max_it = self.settings.max_iterations

    def run(self):
        """Runs the model"""

        if self.settings.transient:
            # Solve transient
            self.progress = tqdm.autonotebook.tqdm(
                desc="Solving H transport problem",
                total=self.settings.final_time,
                unit_scale=True,
            )
            while self.t.value < self.settings.final_time:
                self.iterate()
        else:
            # Solve steady-state
            self.solver.solve(self.u)
            self.post_processing()

    def iterate(self):
        """Iterates the model for a given time step"""
        self.progress.update(
            min(self.dt.value, abs(self.settings.final_time - self.t.value))
        )
        self.t.value += self.dt.value

        self.update_time_dependent_values()

        # solve main problem
        nb_its, converged = self.solver.solve(self.u)

        # post processing
        self.post_processing()

        # update previous solution
        self.u_n.x.array[:] = self.u.x.array[:]

        # adapt stepsize
        if self.settings.stepsize.adaptive:
            new_stepsize = self.settings.stepsize.modify_value(
                value=self.dt.value, nb_iterations=nb_its, t=self.t.value
            )
            self.dt.value = new_stepsize

    def update_time_dependent_values(self):
        t = float(self.t)
        if self.temperature_time_dependent:
            if isinstance(self.temperature_fenics, fem.Constant):
                self.temperature_fenics.value = self.temperature(t=t)
            elif isinstance(self.temperature_fenics, fem.Function):
                self.temperature_fenics.interpolate(self.temperature_expr)
        for bc in self.boundary_conditions:
            if bc.time_dependent:
                bc.update(t=t)
            elif self.temperature_time_dependent and bc.temperature_dependent:
                bc.update(t=t)

        for source in self.sources:
            if source.time_dependent:
                source.update(t=t)
            elif self.temperature_time_dependent and source.temperature_dependent:
                source.update(t=t)

    def post_processing(self):
        """Post processes the model"""

        if self.temperature_time_dependent:
            # update global D if temperature time dependent or internal
            # variables time dependent
            species_not_updated = self.species.copy()  # make a copy of the species
            for export in self.exports:
                if isinstance(export, F.SurfaceFlux):
                    # if the D of the species has not been updated yet
                    if export.field in species_not_updated:
                        export.D.interpolate(export.D_expr)
                        species_not_updated.remove(export.field)

        for export in self.exports:
            # TODO if export type derived quantity
            if isinstance(export, F.SurfaceQuantity):
                export.compute(
                    self.mesh.n,
                    self.ds,
                )
                # update export data
                export.t.append(float(self.t))

                # if filename given write export data to file
                if export.filename is not None:
                    export.write(t=float(self.t))
            elif isinstance(export, F.VolumeQuantity):
                export.compute(self.dx)
                # update export data
                export.t.append(float(self.t))

                # if filename given write export data to file
                if export.filename is not None:
                    export.write(t=float(self.t))
            if isinstance(export, (F.VTXExport, F.XDMFExport)):
                export.write(float(self.t))
