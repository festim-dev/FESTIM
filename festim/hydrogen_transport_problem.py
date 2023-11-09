from dolfinx import fem
from dolfinx.mesh import meshtags
from dolfinx.nls.petsc import NewtonSolver
import basix
import ufl
from mpi4py import MPI
from dolfinx.mesh import meshtags
import numpy as np
import tqdm.autonotebook
import festim as F


class HydrogenTransportProblem:
    """
    Hydrogen Transport Problem.

    Args:
        mesh (festim.Mesh): the mesh of the model
        subdomains (list of festim.Subdomain): the subdomains of the model
        species (list of festim.Species): the species of the model
        temperature (float, int, fem.Constant, fem.Function or callable): the
            temperature of the model (K)
        sources (list of festim.Source): the hydrogen sources of the model
        boundary_conditions (list of festim.BoundaryCondition): the boundary
            conditions of the model
        solver_parameters (dict): the solver parameters of the model
        exports (list of festim.Export): the exports of the model

    Attributes:
        mesh (festim.Mesh): the mesh of the model
        subdomains (list of festim.Subdomain): the subdomains of the model
        species (list of festim.Species): the species of the model
        temperature (float, int, fem.Constant, fem.Function or callable): the
            temperature of the model (K)
        boundary_conditions (list of festim.BoundaryCondition): the boundary
            conditions of the model
        solver_parameters (dict): the solver parameters of the model
        exports (list of festim.Export): the exports of the model
        dx (dolfinx.fem.dx): the volume measure of the model
        ds (dolfinx.fem.ds): the surface measure of the model
        function_space (dolfinx.fem.FunctionSpace): the function space of the
            model
        facet_meshtags (dolfinx.mesh.MeshTags): the facet tags of the model
        volume_meshtags (dolfinx.mesh.MeshTags): the volume tags of the
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
        V_DG_0 (dolfinx.fem.FunctionSpace): A DG function space of degree 0
            over domain
        V_DG_1 (dolfinx.fem.FunctionSpace): A DG function space of degree 1
            over domain


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
        settings=None,
        exports=[],
    ) -> None:
        self.mesh = mesh
        self.subdomains = subdomains
        self.species = species
        self.temperature = temperature
        self.sources = sources
        self.boundary_conditions = boundary_conditions
        self.settings = settings
        self.exports = exports

        self.dx = None
        self.ds = None
        self.function_space = None
        self.facet_meshtags = None
        self.volume_meshtags = None
        self.formulation = None
        self.volume_subdomains = []
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
        elif not isinstance(value, (fem.Constant, fem.Function)):
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

    def initialise(self):
        self.define_function_spaces()
        self.define_markers_and_measures()
        self.assign_functions_to_species()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        self.dt = self.settings.stepsize.get_dt(self.mesh.mesh)

        self.define_temperature()
        self.define_boundary_conditions()
        self.define_sources()
        self.create_formulation()
        self.create_solver()
        self.initialise_exports()

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
                function_space_temperature = fem.FunctionSpace(
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
                    # TODO check if mobile or immobile for traps
                    elements.append(element_CG)
            element = ufl.MixedElement(elements)

        self.function_space = fem.FunctionSpace(self.mesh.mesh, element)

        # create global DG function spaces of degree 0 and 1
        self.V_DG_0 = fem.FunctionSpace(self.mesh.mesh, ("DG", 0))
        self.V_DG_1 = fem.FunctionSpace(self.mesh.mesh, ("DG", 1))

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

    def define_markers_and_measures(self):
        """Defines the markers and measures of the model"""

        facet_indices, tags_facets = [], []

        # find all cells in domain and mark them as 0
        num_cells = self.mesh.mesh.topology.index_map(self.mesh.vdim).size_local
        mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
        tags_volumes = np.full(num_cells, 0, dtype=np.int32)

        for sub_dom in self.subdomains:
            if isinstance(sub_dom, F.SurfaceSubdomain1D):
                facet_index = sub_dom.locate_boundary_facet_indices(
                    self.mesh.mesh, self.mesh.fdim
                )
                facet_indices.append(facet_index)
                tags_facets.append(sub_dom.id)
            if isinstance(sub_dom, F.VolumeSubdomain1D):
                # find all cells in subdomain and mark them as sub_dom.id
                self.volume_subdomains.append(sub_dom)
                entities = sub_dom.locate_subdomain_entities(
                    self.mesh.mesh, self.mesh.vdim
                )
                tags_volumes[entities] = sub_dom.id

        # check if all borders are defined
        if isinstance(self.mesh, F.Mesh1D):
            self.mesh.check_borders(self.volume_subdomains)

        # dofs and tags need to be in np.in32 format for meshtags
        facet_indices = np.array(facet_indices, dtype=np.int32)
        tags_facets = np.array(tags_facets, dtype=np.int32)

        # define mesh tags
        self.facet_meshtags = meshtags(
            self.mesh.mesh, self.mesh.fdim, facet_indices, tags_facets
        )
        self.volume_meshtags = meshtags(
            self.mesh.mesh, self.mesh.vdim, mesh_cell_indices, tags_volumes
        )

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

    def define_sources(self):
        """Create a fenics object value for each source of the model"""
        for source in self.sources:
            for idx, spe in enumerate(source.species):
                if isinstance(spe, str):
                    # if name of species is given then replace with species object
                    source.species[idx] = F.find_species_from_name(spe, self.species)
            for idx, vol in enumerate(source.volume):
                if isinstance(vol, int):
                    # if name of species is given then replace with species object
                    source.volume[idx] = F.find_volume_from_id(
                        source.volume, self.volume_subdomains
                    )
            if isinstance(source, F.Source):
                function_space_value = None
                if callable(source.value):
                    # if bc.value is a callable then need to provide a functionspace
                    if not self.multispecies:
                        function_space_value = source.species[0].sub_function_space
                    else:
                        function_space_value = source.species[
                            0
                        ].collapsed_function_space

                source.create_value(
                    mesh=self.mesh.mesh,
                    temperature=self.temperature_fenics,
                    function_space=function_space_value,
                    t=self.t,
                )

    def create_formulation(self):
        """Creates the formulation of the model"""
        if len(self.sources) > 1:
            raise NotImplementedError("Sources not implemented yet")

        self.formulation = 0

        for spe in self.species:
            u = spe.solution
            u_n = spe.prev_solution
            v = spe.test_function

            for vol in self.volume_subdomains:
                D = vol.material.get_diffusion_coefficient(
                    self.mesh.mesh, self.temperature_fenics, spe
                )

                self.formulation += ufl.dot(D * ufl.grad(u), ufl.grad(v)) * self.dx(
                    vol.id
                )
                self.formulation += ((u - u_n) / self.dt) * v * self.dx(vol.id)

                # add sources
                for source in self.sources:
                    for source_spe in source.species:
                        for source_vol in source.volume:
                            if source_spe == spe and source_vol == vol:
                                self.formulation -= (
                                    source.value_fenics * v * self.dx(vol.id)
                                )

                # add fluxes
                # TODO implement this
                # for bc in self.boundary_conditions:
                #     pass
                #     if bc.species == spe and bc.type != "dirichlet":
                #         formulation += bc * v * self.ds

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

        self.progress = tqdm.autonotebook.tqdm(
            desc="Solving H transport problem",
            total=self.settings.final_time,
            unit_scale=True,
        )
        while self.t.value < self.settings.final_time:
            self.iterate()

    def iterate(self):
        """Iterates the model for a given time step"""
        self.progress.update(self.dt.value)
        self.t.value += self.dt.value

        self.update_time_dependent_values()

        # solve main problem
        self.solver.solve(self.u)

        # post processing
        self.post_processing()

        # update previous solution
        self.u_n.x.array[:] = self.u.x.array[:]

    def update_time_dependent_values(self):
        t = float(self.t)
        if self.temperature_time_dependent:
            if isinstance(self.temperature_fenics, fem.Constant):
                self.temperature_fenics.value = self.temperature(t=t)
            else:
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

            if isinstance(export, (F.VTXExport, F.XDMFExport)):
                export.write(float(self.t))
