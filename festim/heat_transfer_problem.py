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


class HeatTransferProblem:
    def __init__(
        self,
        mesh=None,
        subdomains=None,
        initial_conditions=None,
        boundary_conditions=None,
        sources=None,
        exports=None,
    ) -> None:
        self.mesh = mesh
        self.subdomains = subdomains or []
        self.initial_conditions = initial_conditions or []
        self.boundary_conditions = boundary_conditions or []
        self.sources = sources or []
        self.exports = exports or []

        self.dx = None
        self.ds = None
        self.function_space = None
        self.facet_meshtags = None
        self.volume_meshtags = None
        self.formulation = None
        self.bc_forms = []

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, value):
        if not all(isinstance(source, F.HeatSource) for source in value):
            raise TypeError("sources must be a list of festim.HeatSource objects")
        self._sources = value

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, value):
        if not all(isinstance(bc, F.FixedTemperatureBC) for bc in value):
            raise TypeError(
                "boundary_conditions must be a list of festim.FixedTemperatureBC objects"
            )
        self._boundary_conditions = value

    def initialise(self):
        self.define_function_space()
        self.define_meshtags_and_measures()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self.dt = F.as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )

        self.define_boundary_conditions()
        self.create_source_values_fenics()
        self.create_initial_conditions()
        self.create_formulation()
        self.create_solver()
        self.initialise_exports()

    def define_function_space(self):
        degree = 1
        element_CG = basix.ufl.element(
            basix.ElementFamily.P,
            self.mesh.mesh.basix_cell(),
            degree,
            basix.LagrangeVariant.equispaced,
        )
        self.function_space = fem.functionspace(self.mesh.mesh, element_CG)

        self.u = fem.Function(self.function_space)
        self.u_n = fem.Function(self.function_space)
        self.test_function = ufl.TestFunction(self.function_space)

    # TODO this is very identical to HydrogenTransportProblem
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
            # FIXME # refer to issue #647
            # create empty mesh tags for now
            facet_indices = np.array([], dtype=np.int32)
            facet_tags = np.array([], dtype=np.int32)
            self.facet_meshtags = meshtags(
                self.mesh.mesh, self.mesh.fdim, facet_indices, facet_tags
            )

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
            if isinstance(bc, F.DirichletBCBase):
                form = self.create_dirichletbc_form(bc)
                self.bc_forms.append(form)

    def create_dirichletbc_form(self, bc):
        """Creates a dirichlet boundary condition form

        Args:
            bc (festim.FixedTemperatureBC): the boundary condition

        Returns:
            dolfinx.fem.bcs.DirichletBC: A representation of
                the boundary condition for modifying linear systems.
        """
        bc.create_value(
            mesh=self.mesh.mesh,
            function_space=self.function_space,
            t=self.t,
        )

        bc_dofs = bc.define_surface_subdomain_dofs(
            facet_meshtags=self.facet_meshtags,
            mesh=self.mesh,
            function_space=self.function_space,
        )

        if isinstance(bc.value_fenics, (fem.Function)):
            form = fem.dirichletbc(value=bc.value_fenics, dofs=bc_dofs)
        else:
            form = fem.dirichletbc(
                value=bc.value_fenics, dofs=bc_dofs, V=self.function_space
            )

        return form

    def create_source_values_fenics(self):
        """For each source create the value_fenics"""
        for source in self.sources:
            # create value_fenics for all source objects
            source.create_value_fenics(
                mesh=self.mesh.mesh,
                function_space=self.function_space,
                t=self.t,
            )

    def create_initial_conditions(self):
        """For each initial condition, create the value_fenics and assign it to
        the previous solution of the condition's species"""

        if len(self.initial_conditions) > 0 and not self.settings.transient:
            raise ValueError(
                "Initial conditions can only be defined for transient simulations"
            )

        for condition in self.initial_conditions:
            raise NotImplementedError(
                "Initial conditions are not implemented yet for heat transfer problems"
            )
            # create value_fenics for condition

            condition.create_expr_fenics(
                mesh=self.mesh.mesh,
                temperature=None,
                function_space=self.function_space,
            )

            # assign to previous solution of species

            self.u_n.interpolate(condition.expr_fenics)

    def create_formulation(self):
        """Creates the formulation of the model"""

        self.formulation = 0

        # add diffusion and time derivative for each species
        for vol in self.volume_subdomains:
            thermal_cond = vol.material.thermal_conductivity
            if callable(thermal_cond):
                thermal_cond = thermal_cond(self.u)

            self.formulation += ufl.dot(
                thermal_cond * ufl.grad(self.u), ufl.grad(self.test_function)
            ) * self.dx(vol.id)

            if self.settings.transient:
                raise NotImplementedError(
                    "Transient simulations are not implemented yet, need to add rho and cp"
                )
                self.formulation += (
                    rho
                    * cp
                    * ((self.u - self.u_n) / self.dt)
                    * self.test_function
                    * self.dx(vol.id)
                )

        # add sources
        for source in self.sources:
            self.formulation -= (
                source.value_fenics * self.test_function * self.dx(source.volume.id)
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

    def initialise_exports(self):
        """Defines the export writers of the model, if field is given as
        a string, find species object in self.species"""

        for export in self.exports:
            if isinstance(export, (F.VTXExport, F.XDMFExport)):
                raise NotImplementedError(
                    "VTX and XDMF exports are not implemented yet for heat transfer problems"
                )
                export.define_writer(MPI.COMM_WORLD)
                if isinstance(export, F.XDMFExport):
                    export.writer.write_mesh(self.mesh.mesh)

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
        self.progress.update(self.dt.value)
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
        for bc in self.boundary_conditions:
            if bc.time_dependent:
                bc.update(t=t)

        for source in self.sources:
            if source.time_dependent:
                source.update(t=t)

    def post_processing(self):
        """Post processes the model"""

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
