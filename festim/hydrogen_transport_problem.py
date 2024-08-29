import dolfinx
from dolfinx import fem
import basix
import ufl
from mpi4py import MPI
import numpy as np
import tqdm.autonotebook
import festim as F
from festim.helpers_discontinuity import NewtonSolver


class HydrogenTransportProblem(F.ProblemBase):
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
        >>> my_model.sources = [F.ParticleSource(...)]
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
        super().__init__(
            mesh=mesh,
            sources=sources,
            exports=exports,
            subdomains=subdomains,
            boundary_conditions=boundary_conditions,
            settings=settings,
        )

        self.species = species or []
        self.temperature = temperature
        self.reactions = reactions or []
        self.initial_conditions = initial_conditions or []
        self.traps = traps or []
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
        self.create_flux_values_fenics()
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

        # TODO: expose degree as a property to the user (element_degree ?) in ProblemBase
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

    def define_boundary_conditions(self):
        for bc in self.boundary_conditions:
            if isinstance(bc.species, str):
                # if name of species is given then replace with species object
                bc.species = F.find_species_from_name(bc.species, self.species)

        super().define_boundary_conditions()

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
            # create value_fenics for all F.ParticleSource objects
            if isinstance(source, F.ParticleSource):
                source.create_value_fenics(
                    mesh=self.mesh.mesh,
                    temperature=self.temperature_fenics,
                    t=self.t,
                )

    def create_flux_values_fenics(self):
        """For each particle flux create the value_fenics"""
        for bc in self.boundary_conditions:
            # create value_fenics for all F.ParticleFluxBC objects
            if isinstance(bc, F.ParticleFluxBC):
                bc.create_value_fenics(
                    mesh=self.mesh.mesh,
                    temperature=self.temperature_fenics,
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
            for reactant in reaction.reactant:
                if isinstance(reactant, F.Species):
                    self.formulation += (
                        reaction.reaction_term(self.temperature_fenics)
                        * reactant.test_function
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
        for bc in self.boundary_conditions:
            if isinstance(bc, F.ParticleFluxBC):
                self.formulation -= (
                    bc.value_fenics
                    * bc.species.test_function
                    * self.ds(bc.subdomain.id)
                )

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

    def update_time_dependent_values(self):
        super().update_time_dependent_values()

        if not self.temperature_time_dependent:
            return

        t = float(self.t)

        if isinstance(self.temperature_fenics, fem.Constant):
            self.temperature_fenics.value = self.temperature(t=t)
        elif isinstance(self.temperature_fenics, fem.Function):
            self.temperature_fenics.interpolate(self.temperature_expr)

        for bc in self.boundary_conditions:
            if bc.temperature_dependent:
                bc.update(t=t)

        for source in self.sources:
            if source.temperature_dependent:
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
                if isinstance(
                    export, (F.SurfaceFlux, F.TotalSurface, F.AverageSurface)
                ):
                    export.compute(
                        self.ds,
                    )
                else:
                    export.compute()
                # update export data
                export.t.append(float(self.t))

                # if filename given write export data to file
                if export.filename is not None:
                    export.write(t=float(self.t))
            elif isinstance(export, F.VolumeQuantity):
                if isinstance(export, (F.TotalVolume, F.AverageVolume)):
                    export.compute(self.dx)
                else:
                    export.compute()
                # update export data
                export.t.append(float(self.t))

                # if filename given write export data to file
                if export.filename is not None:
                    export.write(t=float(self.t))
            if isinstance(export, (F.VTXExport, F.XDMFExport)):
                export.write(float(self.t))


class HTransportProblemDiscontinuous(HydrogenTransportProblem):

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
        interfaces=None,
        surface_to_volume=None,
    ):
        super().__init__(
            mesh,
            subdomains,
            species,
            reactions,
            temperature,
            sources,
            initial_conditions,
            boundary_conditions,
            settings,
            exports,
            traps,
        )
        self.interfaces = interfaces or {}
        self.surface_to_volume = surface_to_volume or {}

    def initialise(self):
        self.define_meshtags_and_measures()
        self.create_submeshes()
        self.create_species_from_traps()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self.dt = F.as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )
        self.define_temperature()

        self.create_source_values_fenics()
        self.create_flux_values_fenics()
        self.create_initial_conditions()

        for subdomain in self.volume_subdomains:
            self.define_function_spaces(subdomain)
            ct_r = dolfinx.mesh.meshtags(
                self.mesh.mesh,
                self.mesh.mesh.topology.dim,
                subdomain.submesh_to_mesh,
                np.full_like(subdomain.submesh_to_mesh, 1, dtype=np.int32),
            )
            subdomain.dx = ufl.Measure(
                "dx", domain=self.mesh.mesh, subdomain_data=ct_r, subdomain_id=1
            )
            self.create_subdomain_formulation(subdomain)
            subdomain.u.name = f"u_{subdomain.id}"

        self.define_boundary_conditions()

        self.create_formulation()
        self.create_solver()
        self.initialise_exports()

    def create_dirichletbc_form(self, bc):
        fdim = self.mesh.mesh.topology.dim - 1
        volume_subdomain = self.surface_to_volume[bc.subdomain]
        sub_V = bc.species.subdomain_to_function_space[volume_subdomain]
        collapsed_V, _ = sub_V.collapse()

        bc_function = dolfinx.fem.Function(collapsed_V)
        bc_function.x.array[:] = bc.value
        volume_subdomain.submesh.topology.create_connectivity(
            volume_subdomain.submesh.topology.dim - 1,
            volume_subdomain.submesh.topology.dim,
        )
        bc_dofs = dolfinx.fem.locate_dofs_topological(
            (sub_V, collapsed_V),
            fdim,
            volume_subdomain.ft.find(bc.subdomain.id),
        )
        form = dolfinx.fem.dirichletbc(bc_function, bc_dofs, sub_V)
        return form

    def create_initial_conditions(self):
        if self.initial_conditions:
            raise NotImplementedError(
                "initial conditions not yet implemented for discontinuous"
            )

    def create_submeshes(self):

        for subdomain in self.volume_subdomains:
            subdomain.create_subdomain(self.mesh.mesh, self.volume_meshtags)
            subdomain.transfer_meshtag(self.mesh.mesh, self.facet_meshtags)

    def define_function_spaces(self, subdomain: F.VolumeSubdomain):
        # get number of species defined in the subdomain
        all_species = [
            species for species in self.species if subdomain in species.subdomains
        ]

        # instead of using the set function we use a list to keep the order
        unique_species = []
        for species in all_species:
            if species not in unique_species:
                unique_species.append(species)
        nb_species = len(unique_species)

        degree = 1
        element_CG = basix.ufl.element(
            basix.ElementFamily.P,
            subdomain.submesh.basix_cell(),
            degree,
            basix.LagrangeVariant.equispaced,
        )
        element = basix.ufl.mixed_element([element_CG] * nb_species)
        V = dolfinx.fem.functionspace(subdomain.submesh, element)
        u = dolfinx.fem.Function(V)
        u_n = dolfinx.fem.Function(V)

        us = list(ufl.split(u))
        u_ns = list(ufl.split(u_n))
        vs = list(ufl.TestFunctions(V))
        for i, species in enumerate(unique_species):
            species.subdomain_to_solution[subdomain] = us[i]
            species.subdomain_to_prev_solution[subdomain] = u_ns[i]
            species.subdomain_to_test_function[subdomain] = vs[i]
            species.subdomain_to_function_space[subdomain] = V.sub(i)
            species.subdomain_to_post_processing_solution[subdomain] = u.sub(
                i
            ).collapse()
            species.subdomain_to_collapsed_function_space[subdomain] = V.sub(
                i
            ).collapse()
            species.subdomain_to_post_processing_solution[subdomain].name = (
                f"{species.name}_{subdomain.id}"
            )
        subdomain.u = u
        subdomain.u_n = u_n

    def create_subdomain_formulation(self, subdomain: F.VolumeSubdomain):
        form = 0
        # add diffusion and time derivative for each species
        for spe in self.species:
            if subdomain not in spe.subdomains:
                continue
            u = spe.subdomain_to_solution[subdomain]
            u_n = spe.subdomain_to_prev_solution[subdomain]
            v = spe.subdomain_to_test_function[subdomain]
            dx = subdomain.dx

            D = subdomain.material.get_diffusion_coefficient(
                self.mesh.mesh, self.temperature_fenics, spe
            )
            if self.settings.transient:
                form += ((u - u_n) / self.dt) * v * dx

            if spe.mobile:
                form += ufl.inner(D * ufl.grad(u), ufl.grad(v)) * dx

        for reaction in self.reactions:
            if reaction.volume != subdomain:
                continue
            for species in reaction.reactant + reaction.product:
                if isinstance(species, F.Species):
                    # TODO remove
                    # temporarily overide the solution to the one of the subdomain
                    species.solution = species.subdomain_to_solution[subdomain]

            for reactant in reaction.reactant:
                if isinstance(reactant, F.Species):
                    form += (
                        reaction.reaction_term(self.temperature_fenics)
                        * reactant.subdomain_to_test_function[subdomain]
                        * dx
                    )

            # product
            if isinstance(reaction.product, list):
                products = reaction.product
            else:
                products = [reaction.product]
            for product in products:
                form += (
                    -reaction.reaction_term(self.temperature_fenics)
                    * product.subdomain_to_test_function[subdomain]
                    * dx
                )

        # add fluxes
        for bc in self.boundary_conditions:
            if isinstance(bc, F.ParticleFluxBC):
                if subdomain == self.surface_to_volume[bc.subdomain]:
                    v = bc.species.subdomain_to_test_function[subdomain]
                    form -= bc.value_fenics * v * self.ds(bc.subdomain.id)

        subdomain.F = form

    def create_formulation(self):
        # Add coupling term to the interface
        # Get interface markers on submesh b
        mesh = self.mesh.mesh
        ct = self.volume_meshtags
        mt = self.facet_meshtags
        f_to_c = mesh.topology.connectivity(mesh.topology.dim - 1, mesh.topology.dim)

        for interface in self.interfaces:
            interface.mesh = mesh
            interface.mt = mt

        integral_data = [
            interface.compute_mapped_interior_facet_data(mesh)
            for interface in self.interfaces
        ]
        [interface.pad_parent_maps() for interface in self.interfaces]
        dInterface = ufl.Measure("dS", domain=mesh, subdomain_data=integral_data)

        def mixed_term(u, v, n):
            return ufl.dot(ufl.grad(u), n) * v

        n = ufl.FacetNormal(mesh)
        cr = ufl.Circumradius(mesh)

        entity_maps = {
            sd.submesh: sd.parent_to_submesh for sd in self.volume_subdomains
        }
        for interface in self.interfaces:
            gamma = interface.penalty_term

            subdomain_1, subdomain_2 = interface.subdomains
            b_res, t_res = interface.restriction
            n_b = n(b_res)
            n_t = n(t_res)
            h_b = 2 * cr(b_res)
            h_t = 2 * cr(t_res)

            # look at the first facet on interface
            # and get the two cells that are connected to it
            # and get the material properties of these cells
            first_facet_interface = mt.find(interface.id)[0]
            c_plus, c_minus = (
                f_to_c.links(first_facet_interface)[0],
                f_to_c.links(first_facet_interface)[1],
            )
            id_minus, id_plus = ct.values[c_minus], ct.values[c_plus]

            for subdomain in interface.subdomains:
                if subdomain.id == id_plus:
                    subdomain_1 = subdomain
                if subdomain.id == id_minus:
                    subdomain_2 = subdomain

            all_mobile_species = [spe for spe in self.species if spe.mobile]
            if len(all_mobile_species) > 1:
                raise NotImplementedError("Multiple mobile species not implemented")
            H = all_mobile_species[0]

            v_b = H.subdomain_to_test_function[subdomain_1](b_res)
            v_t = H.subdomain_to_test_function[subdomain_2](t_res)

            u_b = H.subdomain_to_solution[subdomain_1](b_res)
            u_t = H.subdomain_to_solution[subdomain_2](t_res)

            K_b = subdomain_1.material.get_solubility_coefficient(
                self.mesh.mesh, self.temperature_fenics(b_res), H
            )
            K_t = subdomain_2.material.get_solubility_coefficient(
                self.mesh.mesh, self.temperature_fenics(t_res), H
            )

            F_0 = -0.5 * mixed_term((u_b + u_t), v_b, n_b) * dInterface(
                interface.id
            ) - 0.5 * mixed_term(v_b, (u_b / K_b - u_t / K_t), n_b) * dInterface(
                interface.id
            )

            F_1 = +0.5 * mixed_term((u_b + u_t), v_t, n_b) * dInterface(
                interface.id
            ) - 0.5 * mixed_term(v_t, (u_b / K_b - u_t / K_t), n_b) * dInterface(
                interface.id
            )
            F_0 += (
                2
                * gamma
                / (h_b + h_t)
                * (u_b / K_b - u_t / K_t)
                * v_b
                * dInterface(interface.id)
            )
            F_1 += (
                -2
                * gamma
                / (h_b + h_t)
                * (u_b / K_b - u_t / K_t)
                * v_t
                * dInterface(interface.id)
            )

            subdomain_1.F += F_0
            subdomain_2.F += F_1

        J = []
        forms = []
        for subdomain1 in self.volume_subdomains:
            jac = []
            form = subdomain1.F
            for subdomain2 in self.volume_subdomains:
                jac.append(
                    dolfinx.fem.form(
                        ufl.derivative(form, subdomain2.u), entity_maps=entity_maps
                    )
                )
            J.append(jac)
            forms.append(dolfinx.fem.form(subdomain1.F, entity_maps=entity_maps))

        self.forms = forms
        self.J = J

    def create_solver(self):
        self.solver = NewtonSolver(
            self.forms,
            self.J,
            [subdomain.u for subdomain in self.volume_subdomains],
            bcs=self.bc_forms,
            max_iterations=10,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )

    def create_flux_values_fenics(self):
        """For each particle flux create the value_fenics"""
        for bc in self.boundary_conditions:
            # create value_fenics for all F.ParticleFluxBC objects
            if isinstance(bc, F.ParticleFluxBC):
                volume_subdomain = self.surface_to_volume[bc.subdomain]
                bc.create_value_fenics(
                    mesh=volume_subdomain.submesh,
                    temperature=self.temperature_fenics,
                    t=self.t,
                )

    def initialise_exports(self):
        for export in self.exports:
            if isinstance(export, F.VTXExport):
                species = export.field[0]
                # override post_processing_solution attribute of species
                species.post_processing_solution = (
                    species.subdomain_to_post_processing_solution[export.subdomain]
                )
                export.define_writer(MPI.COMM_WORLD)
            else:
                raise NotImplementedError("Export type not implemented")

    def post_processing(self):
        for subdomain in self.volume_subdomains:
            for species in self.species:
                if subdomain not in species.subdomains:
                    continue
                collapsed_function = species.subdomain_to_post_processing_solution[
                    subdomain
                ]
                u = subdomain.u
                v0_to_V = species.subdomain_to_collapsed_function_space[subdomain][1]
                collapsed_function.x.array[:] = u.x.array[v0_to_V]

        for export in self.exports:
            if isinstance(export, F.VTXExport):
                export.write(float(self.t))
            else:
                raise NotImplementedError("Export type not implemented")

    def iterate(self):
        """Iterates the model for a given time step"""
        if self.show_progress_bar:
            self.progress_bar.update(
                min(self.dt.value, abs(self.settings.final_time - self.t.value))
            )
        self.t.value += self.dt.value

        self.update_time_dependent_values()

        # solve main problem
        self.solver.solve(1e-5)

        # post processing
        self.post_processing()

        # update previous solution
        for subdomain in self.volume_subdomains:
            subdomain.u_n.x.array[:] = subdomain.u.x.array[:]

        # adapt stepsize
        if self.settings.stepsize.adaptive:
            raise NotImplementedError("Adaptive stepsize not implemented")

    def run(self):
        if self.settings.transient:
            # Solve transient
            if self.show_progress_bar:
                self.progress_bar = tqdm.autonotebook.tqdm(
                    desc=f"Solving {self.__class__.__name__}",
                    total=self.settings.final_time,
                    unit_scale=True,
                )
            while self.t.value < self.settings.final_time:
                self.iterate()
            if self.show_progress_bar:
                self.progress_bar.refresh()  # refresh progress bar to show 100%
        else:
            # Solve steady-state
            self.solver.solve(1e-5)
            self.post_processing()
