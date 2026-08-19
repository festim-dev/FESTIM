import warnings
from collections.abc import Callable

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import io4dolfinx
import numpy as np
import numpy.typing as npt
import scifem
import tqdm.auto
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from packaging.version import Version

from festim import (
    boundary_conditions,
    exports,
    k_B,
    problem,
)
from festim import (
    reaction as _reaction,
)
from festim import source as _source
from festim import (
    species as _species,
)
from festim import (
    subdomain as _subdomain,
)
from festim.advection import AdvectionTerm
from festim.enclosure._utils import check_dolfinx_version_for_enclosures
from festim.enclosure.enclosure import Enclosure as _Enclosure
from festim.enclosure.gas_species import GasSpecies as _GasSpecies
from festim.enclosure.gas_species import create_real_function_space
from festim.enclosure.openings import EnclosureConnection
from festim.helpers import (
    KSPMonitor,
    SnesMonitor,
    as_fenics_constant,
    convergenceTest,
    is_it_time_to_export,
    nmm_interpolate,
)

from .mesh import CoordinateSystem, Mesh

__all__ = [
    "HydrogenTransportProblem",
    "HydrogenTransportProblemDG",
    "HydrogenTransportProblemDiscontinuous",
]


class HydrogenTransportProblem(problem.ProblemBase):
    """Hydrogen Transport Problem.

    Args:
        mesh: The mesh
        subdomains: list containing the subdomains
        species: list containing the species
        reactions: list containing the reactions
        temperature: The temperature or a function describing the temperature as
            a model of either space or space and time. Unit (K)
        sources: The hydrogen sources
        initial_conditions: The initial conditions
        boundary_conditions: The boundary conditions
        exports (list of festim.Export): the exports of the model
        traps (list of F.Trap): the traps of the model
        advection_terms: the advection terms of the model

    Attributes:
        mesh : The mesh
        subdomains: The subdomains
        species: The species
        reactions: the reaction
        temperature: The temperature in unit `K`
        sources: The hydrogen sources
        initial_conditions: The initial conditions
        boundary_conditions: list of Dirichlet boundary conditions
        exports (list of festim.Export): the export
        traps (list of F.Trap): the traps of the model
        advection_terms: the advection terms of the model
        dx (dolfinx.fem.dx): the volume measure of the model
        ds (dolfinx.fem.ds): the surface measure of the model
        function_space (dolfinx.fem.FunctionSpaceBase): the function space of the
            model
        facet_meshtags (dolfinx.mesh.MeshTags): the facet meshtags of the model
        volume_meshtags (dolfinx.mesh.MeshTags): the volume meshtags of the
            model
        formulation (ufl.form.Form): the formulation of the model
        solver (dolfinx.nls.newton.NewtonSolver): the solver of the model
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


    Examples:
        Can be used as either

        .. highlight:: python
        .. code-block:: python

            import festim as F
            my_model = F.HydrogenTransportProblem()
            my_model.mesh = F.Mesh(...)
            my_model.subdomains = [F.Subdomain(...)]
            my_model.species = [F.Species(name="H"), F.Species(name="Trap")]
            my_model.temperature = 500
            my_model.sources = [F.ParticleSource(...)]
            my_model.boundary_conditions = [F.BoundaryCondition(...)]
            my_model.initialise()

        or

        .. highlight:: python
        .. code-block:: python

            my_model = F.HydrogenTransportProblem(
                mesh=F.Mesh(...),
                subdomains=[F.Subdomain(...)],
                species=[F.Species(name="H"), F.Species(name="Trap")],
            )
            my_model.initialise()
    """

    _temperature_as_function: fem.Function
    _species_to_D_global: dict[_species.Species, fem.Function]
    _species_to_D_global_expr: dict[_species.Species, fem.Expression]

    def __init__(
        self,
        mesh: Mesh | None = None,
        subdomains: (
            list[_subdomain.VolumeSubdomain | _subdomain.SurfaceSubdomain] | None
        ) = None,
        species: list[_species.Species] | None = None,
        reactions: list[_reaction.ReactionBase] | None = None,
        temperature: (
            float
            | int
            | fem.Constant
            | fem.Function
            | Callable[
                [npt.NDArray[dolfinx.default_scalar_type]],  # type: ignore
                npt.NDArray[dolfinx.default_scalar_type],  # type: ignore
            ]
            | Callable[
                [npt.NDArray[dolfinx.default_scalar_type], fem.Constant],  # type: ignore
                npt.NDArray[dolfinx.default_scalar_type],  # type: ignore
            ]
            | None
        ) = None,
        sources=None,
        initial_conditions=None,
        boundary_conditions=None,
        settings=None,
        exports=None,
        traps=None,
        advection_terms=None,
        petsc_options=None,
        element_immobile: str = "CG",
    ):
        super().__init__(
            mesh=mesh,
            sources=sources,
            exports=exports,
            subdomains=subdomains,
            boundary_conditions=boundary_conditions,
            settings=settings,
            petsc_options=petsc_options,
        )

        self.species = species or []
        self.temperature = temperature
        self.reactions = reactions or []
        self.initial_conditions = initial_conditions or []
        self.traps = traps or []
        self.advection_terms = advection_terms or []
        self.temperature_fenics = None

        self._unpacked_sources = []

        self._element_immobile = element_immobile

        self._temperature_as_function = None
        self._species_to_D_global = None
        self._species_to_D_global_expr = None
        self._surface_to_volume = None

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value is None:
            self._temperature = value
        elif isinstance(value, float | int | fem.Constant | fem.Function):
            self._temperature = value
        elif callable(value):
            self._temperature = value
        else:
            raise TypeError(
                "Value must be a float, int, fem.Constant, fem.Function, or callable"
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
            fem.Constant | fem.Function,
        ):
            raise TypeError("Value must be a fem.Constant or fem.Function")
        self._temperature_fenics = value

    @property
    def temperature_time_dependent(self):
        if self.temperature is None:
            return False
        if isinstance(self.temperature, fem.Constant | fem.Function):
            return False
        if callable(self.temperature):
            arguments = self.temperature.__code__.co_varnames
            return "t" in arguments
        else:
            return False

    @property
    def species(self) -> list[_species.Species]:
        return self._species

    @species.setter
    def species(self, value):
        # check that all species are of type festim.Species
        for spe in value:
            if not isinstance(spe, _species.Species):
                raise TypeError(
                    f"elements of species must be of type festim.Species not "
                    f"{type(spe)}"
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
            raise TypeError("value must be of type dolfinx.mesh.MeshTags")

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
            raise TypeError("value must be of type dolfinx.mesh.MeshTags")

    @property
    def _unpacked_bcs(self):
        """Returns all boundary conditions, including fluxes from surface reactions."""
        all_boundary_conditions = []
        for bc in self.boundary_conditions:
            if isinstance(bc, boundary_conditions.SurfaceReactionBC):
                all_boundary_conditions.extend(bc.flux_bcs)
            else:
                all_boundary_conditions.append(bc)
        return all_boundary_conditions

    @property
    def element_immobile(self):
        return self._element_immobile

    @element_immobile.setter
    def element_immobile(self, value):
        allowed_values = ["DG", "CG", "P"]
        if value not in allowed_values:
            raise ValueError(f"element_immobile should be in {allowed_values}")
        if value == "P":
            value = "CG"
        self._element_immobile = value

    def initialise(self):
        if getattr(self, "enclosures", None):
            raise NotImplementedError(
                f"Gas enclosures are not supported by {self.__class__.__name__}. Use "
                "festim.HydrogenTransportProblemDiscontinuous instead, which solves a "
                "blocked system the enclosure pressures can be added to."
            )
        self.create_species_from_traps()
        self.define_function_spaces(element_degree=self.settings.element_degree)
        self.define_meshtags_and_measures()
        self.assign_functions_to_species()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self._dt = as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )

        self.create_implicit_species_value_fenics()

        self.define_temperature()
        self.define_boundary_conditions()
        self.convert_reaction_rates_to_fenics_objects()
        self.create_sources_from_reactions()
        self.convert_source_input_values_to_fenics_objects()
        self.convert_advection_term_to_fenics_objects()
        self.create_flux_values_fenics()
        self.create_initial_conditions()
        self.create_formulation()
        self.create_solver()
        self.initialise_exports()

    def create_implicit_species_value_fenics(self):
        """For each implicit species, create the value_fenics."""
        for reaction in self.reactions:
            for spe in reaction.species:
                if isinstance(spe, _species.ImplicitSpecies):
                    spe.create_value_fenics(
                        mesh=self.mesh.mesh,
                        t=self.t,
                    )

    def create_species_from_traps(self):
        """Generate a species and reaction per trap defined in self.traps."""

        for trap in self.traps:
            trap.create_species_and_reaction()
            self.species.append(trap.trapped_concentration)
            self.reactions.append(trap.reaction)

    def define_temperature(self):
        """Sets the value of temperature_fenics_value.

        The type depends on self.temperature. If self.temperature is a function on t
        only, create a fem.Constant. Else, create an dolfinx.fem.Expression (stored in
        self.temperature_expr) to be updated, a dolfinx.fem.Function object is created
        from the Expression (stored in self.temperature_fenics_value). Raise a
        ValueError if temperature is None.
        """
        # check if temperature is None
        if self.temperature is None:
            raise ValueError("the temperature attribute needs to be defined")

        # if temperature is a float or int, create a fem.Constant
        elif isinstance(self.temperature, float | int):
            self.temperature_fenics = as_fenics_constant(
                self.temperature, self.mesh.mesh
            )
        # if temperature is a fem.Constant or function, pass it to temperature_fenics
        elif isinstance(self.temperature, fem.Constant | fem.Function):
            self.temperature_fenics = self.temperature

        # if temperature is callable, process accordingly
        elif callable(self.temperature):
            arguments = self.temperature.__code__.co_varnames
            if "t" in arguments and "x" not in arguments:
                if not isinstance(self.temperature(t=float(self.t)), float | int):
                    raise ValueError(
                        f"self.temperature should return a float or an int, not "
                        f"{type(self.temperature(t=float(self.t)))} "
                    )
                # only t is an argument
                self.temperature_fenics = as_fenics_constant(
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
                self.temperature_fenics.name = "temperature"
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = self.t
                if "x" in arguments:
                    kwargs["x"] = x

                # store the expression of the temperature
                # to update the temperature_fenics later
                self.temperature_expr = fem.Expression(
                    self.temperature(**kwargs),
                    function_space_temperature.element.interpolation_points,
                )
                self.temperature_fenics.interpolate(self.temperature_expr)

    def initialise_exports(self):
        """Defines the export writers of the model, if field is given as a string, find
        species object in self.species."""

        for export in self.exports:
            if isinstance(export, exports.ExportBaseClass):
                if export.times:
                    for time in export.times:
                        if time not in self.settings.stepsize.milestones:
                            msg = "To ensure that the exports data at the desired times"
                            msg += "the values in export.times are added to milestones"
                            warnings.warn(msg)
                            self.settings.stepsize.milestones.append(time)
                    self.settings.stepsize.milestones.sort()

                if isinstance(export, exports.VTXTemperatureExport):
                    self._temperature_as_function = (
                        self._get_temperature_field_as_function()
                    )
                    export.writer = dolfinx.io.VTXWriter(
                        comm=self._temperature_as_function.function_space.mesh.comm,
                        filename=export.filename,
                        output=self._temperature_as_function,
                        engine="BP5",
                    )
                    continue

                elif isinstance(export, exports.VTXSpeciesExport):
                    functions = export.get_functions()
                    if not export._checkpoint:
                        export.writer = dolfinx.io.VTXWriter(
                            comm=functions[0].function_space.mesh.comm,
                            filename=export.filename,
                            output=functions,
                            engine="BP5",
                        )

                    else:
                        io4dolfinx.write_mesh(
                            filename=export.filename,
                            mesh=self.mesh.mesh,
                            backend="adios2",
                        )

                elif isinstance(export, exports.CustomFieldExport):
                    export.function = fem.Function(self.V_CG_1)
                    export.set_dolfinx_expression(
                        temperature=self.temperature_fenics,
                        time=self.t,
                    )

                    export.writer = dolfinx.io.VTXWriter(
                        comm=export.function.function_space.mesh.comm,
                        filename=export.filename,
                        output=export.function,
                        engine="BP5",
                    )
                    continue

            elif isinstance(export, exports.DerivedQuantity):
                # raise not implemented error if the derived quantity don't match the
                # type of mesh eg. SurfaceFlux is used with cylindrical mesh
                if self.mesh.coordinate_system != CoordinateSystem.CARTESIAN:
                    raise NotImplementedError(
                        f"Derived quantity exports are not implemented for "
                        f"{self.mesh.coordinate_system!s} meshes"
                    )

            # if name of species is given then replace with species object
            if hasattr(export, "field"):
                if isinstance(export.field, list):
                    for idx, field in enumerate(export.field):
                        if isinstance(field, str):
                            export.field[idx] = _species.find_species_from_name(
                                field, self.species
                            )
                elif isinstance(export.field, str):
                    export.field = _species.find_species_from_name(
                        export.field, self.species
                    )

            # Initialize XDMFFile for writer
            if isinstance(export, exports.XDMFExport):
                export.define_writer(MPI.COMM_WORLD)

            # clean data for profile1D export
            if isinstance(export, exports.Profile1DExport):
                export.data = []
                export.t = []
        # compute diffusivity function for surface fluxes

        # TODO: probably a better way to handle things would be to follow what's done in
        # https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html
        spe_to_D_global_func_expr = {
            spe: self.define_D_global(spe) for spe in self.species if spe.mobile
        }
        self._species_to_D_global_expr = {
            k: v[1] for k, v in spe_to_D_global_func_expr.items()
        }  # links species to D expression
        self._species_to_D_global = {
            k: v[0] for k, v in spe_to_D_global_func_expr.items()
        }  # links species to global D function

        for export in self.exports:
            if isinstance(export, exports.SurfaceQuantity):
                # add the global D to the export
                export.D = self._species_to_D_global.get(export.field)
                export.D_expr = self._species_to_D_global_expr.get(export.field)
            if isinstance(export, exports.MaximumVolume | exports.MinimumVolume):
                export.volume_meshtags = self.volume_meshtags
            if isinstance(export, exports.MaximumSurface | exports.MinimumSurface):
                export.facet_meshtags = self.facet_meshtags

            # reset the data and time for SurfaceQuantity and VolumeQuantity
            if isinstance(export, exports.DerivedQuantity):
                export.t = []
                export.data = []

            if isinstance(export, exports.CustomQuantity):
                kwargs = {
                    species.name: species.post_processing_solution
                    for species in self.species
                }
                kwargs["n"] = ufl.FacetNormal(self.mesh.mesh)
                kwargs["t"] = self.t
                kwargs["T"] = self.temperature_fenics

                # NOTE we need to change our D_global approach
                D_kwargs = {
                    f"D_{sp.name}": self._species_to_D_global[sp]
                    for sp in self.species
                    if sp.mobile  # immobile species don't have a D_global
                }
                kwargs.update(D_kwargs)
                kwargs["D"] = {
                    sp.name: D_kwargs[f"D_{sp.name}"]
                    for sp in self.species
                    if sp.mobile
                }
                if len(self.species) == 1:
                    if self.species[0].mobile:
                        kwargs["D"] = kwargs[f"D_{self.species[0].name}"]
                kwargs["x"] = ufl.SpatialCoordinate(self.mesh.mesh)
                export.ufl_expr = export.expr(**kwargs)

    def _get_temperature_field_as_function(self) -> dolfinx.fem.Function:
        """Based on the type of the temperature_fenics attribute, converts it as a
        Function to be used in VTX export.

        Returns:
            the temperature field of the simulation
        """
        if isinstance(self.temperature_fenics, fem.Function):
            return self.temperature_fenics
        elif isinstance(self.temperature_fenics, fem.Constant):
            # use existing function space if function already exists
            if self._temperature_as_function is None:
                V = dolfinx.fem.functionspace(self.mesh.mesh, ("P", 1))
            else:
                V = self._temperature_as_function.function_space
            temperature_field = dolfinx.fem.Function(V)
            temperature_expr = fem.Expression(
                self.temperature_fenics,
                V.element.interpolation_points,
            )
            temperature_field.interpolate(temperature_expr)
            return temperature_field

    def volume_subdomain_of_surface(
        self, surface: _subdomain.SurfaceSubdomain
    ) -> _subdomain.VolumeSubdomain:
        """Returns the volume subdomain a surface subdomain belongs to.

        The mapping is deduced from the mesh connectivity and the meshtags, and is
        computed once then cached.

        Args:
            surface: the surface subdomain

        Returns:
            the volume subdomain the surface belongs to

        Raises:
            ValueError: if the surface cannot be mapped to a volume subdomain
        """
        if self._surface_to_volume is None:
            mesh = self.mesh.mesh
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)
            self._surface_to_volume = _subdomain.map_surface_to_volume_subdomains(
                ft=self.facet_meshtags,
                ct=self.volume_meshtags,
                facet_to_cell=mesh.topology.connectivity(tdim - 1, tdim),
                volume_subdomains=self.volume_subdomains,
                surface_subdomains=self.surface_subdomains,
                comm=mesh.comm,
            )

        if surface not in self._surface_to_volume:
            raise ValueError(
                f"Surface subdomain {surface.id} could not be mapped to a volume "
                "subdomain. Check that its id matches a tagged facet of the mesh."
            )
        return self._surface_to_volume[surface]

    def define_D_global(self, species):
        """Defines the global diffusion coefficient for a given species.

        Args:
            species (F.Species): the species

        Returns:
            dolfinx.fem.Function, dolfinx.fem.Expression: the global diffusion
                coefficient and the expression of the global diffusion coefficient
                for a given species
        """
        assert isinstance(species, _species.Species)
        # create global D function
        D = fem.Function(self.V_DG_1)

        # if diffusion coeffient has been given as a function, use that
        if self.volume_subdomains[0].material.D:
            if len(self.volume_subdomains) > 1:
                raise NotImplementedError(
                    "Giving the diffusion coefficient as a function is currently "
                    "only supported for a single volume subdomain case"
                )
            return self.volume_subdomains[0].material.D, None

        D_0 = fem.Function(self.V_DG_0)
        E_D = fem.Function(self.V_DG_0)
        for vol in self.volume_subdomains:
            cell_indices = self.volume_meshtags.find(vol.id)

            # replace values of D_0 and E_D by values from the material
            D_0.x.array[cell_indices] = vol.material.get_D_0(species=species)
            E_D.x.array[cell_indices] = vol.material.get_E_D(species=species)

        expr = D_0 * ufl.exp(
            -E_D / as_fenics_constant(k_B, self.mesh.mesh) / self.temperature_fenics
        )
        D_expr = fem.Expression(expr, self.V_DG_1.element.interpolation_points)
        D.interpolate(D_expr)
        return D, D_expr

    def define_function_spaces(self, element_degree: int = 1):
        """Creates the function space of the modelw with a mixed element. Creates the
        main solution and previous solution function u and u_n. Create global DG
        function spaces of degree 0 and 1 for the global diffusion coefficient.

        Args:
            element_degree: Degree order for finite element. Defaults to 1.
        """

        element_CG = basix.ufl.element(
            basix.ElementFamily.P,
            self.mesh.mesh.basix_cell(),
            element_degree,
            basix.LagrangeVariant.equispaced,
        )
        element_DG = basix.ufl.element(
            "DG",
            self.mesh.mesh.basix_cell(),
            element_degree,
            basix.LagrangeVariant.equispaced,
        )

        elements = []
        for spe in self.species:
            if isinstance(spe, _species.Species):
                if spe.mobile:
                    elements.append(element_CG)
                else:
                    match self._element_immobile:
                        case "DG":
                            elements.append(element_DG)
                        case "CG":
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
        self.V_CG_1 = fem.functionspace(self.mesh.mesh, ("CG", 1))

        self.u = fem.Function(self.function_space)
        self.u_n = fem.Function(self.function_space)

    def assign_functions_to_species(self):
        """Creates the solution, prev solution, test function and post-processing
        solution for each species, as well as a collapsed function space for each
        species."""

        sub_solutions = list(ufl.split(self.u))
        sub_prev_solution = list(ufl.split(self.u_n))
        sub_test_functions = list(ufl.TestFunctions(self.function_space))

        for idx, spe in enumerate(self.species):
            spe.sub_function_space = self.function_space.sub(idx)
            spe.sub_function = self.u.sub(idx)  # TODO add this to discontinuous class
            spe.post_processing_solution = self.u.sub(idx).collapse()
            spe.post_processing_solution.name = spe.name
            spe.collapsed_function_space, spe.map_sub_to_main_solution = (
                self.function_space.sub(idx).collapse()
            )

        for idx, spe in enumerate(self.species):
            spe.solution = sub_solutions[idx]
            spe.prev_solution = sub_prev_solution[idx]
            spe.test_function = sub_test_functions[idx]

    def define_boundary_conditions(self):
        """Defines the boundary conditions of the model."""

        for bc in self._unpacked_bcs:
            if isinstance(bc.species, str):
                # if name of species is given then replace with species object
                bc.species = _species.find_species_from_name(bc.species, self.species)
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                bc.create_value_fenics(
                    mesh=self.mesh.mesh,
                    temperature=self.temperature_fenics,
                    t=self.t,
                )

        super().define_boundary_conditions()

    def create_dirichletbc_form(self, bc):
        """Creates a dirichlet boundary condition form.

        Args:
            bc (festim.DirichletBC): the boundary condition

        Returns:
            dolfinx.fem.bcs.DirichletBC: A representation of
                the boundary condition for modifying linear systems.
        """
        # create value_fenics
        bc.create_value(
            temperature=self.temperature_fenics,
            function_space=bc.species.collapsed_function_space,
            t=self.t,
        )

        # get dofs
        if isinstance(bc.value_fenics, (fem.Function)):
            function_space_dofs = (
                bc.species.sub_function_space,
                bc.species.collapsed_function_space,
            )
        else:
            function_space_dofs = bc.species.sub_function_space

        bc_dofs = bc.define_surface_subdomain_dofs(
            facet_meshtags=self.facet_meshtags,
            function_space=function_space_dofs,
        )

        # create form
        form = fem.dirichletbc(
            value=bc.value_fenics,
            dofs=bc_dofs,
            V=bc.species.sub_function_space,
        )

        return form

    def convert_source_input_values_to_fenics_objects(self):
        """For each source create the value_fenics."""
        for source in self._unpacked_sources:
            # create value_fenics for all F.ParticleSource objects
            if isinstance(source, _source.ParticleSource):
                source.value.convert_input_value(
                    function_space=self.function_space,
                    t=self.t,
                    temperature=self.temperature_fenics,
                    up_to_ufl_expr=True,
                )

    def convert_reaction_rates_to_fenics_objects(self):
        """For each reaction convert its rate coefficients to fenics objects."""
        for reaction in self.reactions:
            for rate in reaction.rate_coefficients:
                if rate.input_value is not None:
                    rate.convert_input_value(
                        function_space=getattr(self, "function_space", None),
                        t=self.t,
                        temperature=self.temperature_fenics,
                        subdomain=reaction.volume,
                        up_to_ufl_expr=True,
                    )

    def create_sources_from_reactions(self):
        """Populate _unpacked_sources with the user-provided sources plus one
        volumetric particle source per species participating in each reaction."""
        self._unpacked_sources = list(self.sources)
        for reaction in self.reactions:
            self._unpacked_sources += reaction.create_sources()

    def convert_advection_term_to_fenics_objects(self):
        """For each advection term convert the input value."""

        for advec_term in self.advection_terms:
            advec_term.velocity.convert_input_value(
                function_space=self.function_space, t=self.t
            )

    def create_flux_values_fenics(self):
        """For each particle flux create the value_fenics."""
        for bc in self.boundary_conditions:
            # create value_fenics for all F.ParticleFluxBC objects
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                bc.create_value_fenics(
                    mesh=self.mesh.mesh,
                    temperature=self.temperature_fenics,
                    t=self.t,
                )

    def create_initial_conditions(self):
        """For each initial condition, create the value_fenics and assign it to the
        previous solution of the condition's species."""

        if len(self.initial_conditions) > 0 and not self.settings.transient:
            raise ValueError(
                "Initial conditions can only be defined for transient simulations"
            )

        for condition in self.initial_conditions:
            function_space_value = None
            if callable(condition.value):
                # if bc.value is a callable then need to provide a functionspace
                function_space_value = condition.species.collapsed_function_space

            condition.create_expr_fenics(
                mesh=self.mesh.mesh,
                temperature=self.temperature_fenics,
                function_space=function_space_value,
            )

            # assign to previous solution of species
            entities = self.volume_meshtags.find(condition.volume.id)
            idx = self.species.index(condition.species)
            function_to_interpolate_on = self.u_n.sub(idx)

            function_to_interpolate_on.interpolate(
                condition.expr_fenics, cells0=entities
            )

    def create_formulation(self):
        """Creates the formulation of the model."""

        self.formulation = 0

        # add diffusion and time derivative for each species
        for spe in self.species:
            u = spe.solution
            u_n = spe.prev_solution
            v = spe.test_function

            for vol in self.volume_subdomains:
                if spe.mobile:
                    D = vol.material.get_diffusion_coefficient(
                        self.mesh.mesh, self.temperature_fenics, spe
                    )
                    match self.mesh.coordinate_system:
                        case CoordinateSystem.CARTESIAN:
                            self.formulation += ufl.dot(
                                D * ufl.grad(u), ufl.grad(v)
                            ) * self.dx(vol.id)
                        case CoordinateSystem.CYLINDRICAL:
                            r = ufl.SpatialCoordinate(self.mesh.mesh)[0]
                            self.formulation += (
                                r
                                * ufl.dot(D * ufl.grad(u), ufl.grad(v / r))
                                * self.dx(vol.id)
                            )
                        case CoordinateSystem.SPHERICAL:
                            r = ufl.SpatialCoordinate(self.mesh.mesh)[0]
                            self.formulation += (
                                r**2
                                * ufl.dot(D * ufl.grad(u), ufl.grad(v / r**2))
                                * self.dx(vol.id)
                            )
                        case _:
                            raise NotImplementedError(
                                f"Unknown coordinate system {self.mesh.coordinate_system!s}"  # noqa: E501
                            )

                if self.settings.transient:
                    self.formulation += ((u - u_n) / self.dt) * v * self.dx(vol.id)

        for source in self._unpacked_sources:
            self.formulation -= (
                source.value.fenics_object
                * source.species.test_function
                * self.dx(source.volume.id)
            )

        # add boundary conditions (fluxes and weak dirichlet)
        for bc in self.boundary_conditions:
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                self.formulation -= (
                    bc.value_fenics
                    * bc.species.test_function
                    * self.ds(bc.subdomain.id)
                )
            if isinstance(bc, boundary_conditions.SurfaceReactionBC):
                for flux_bc in bc.flux_bcs:
                    self.formulation -= (
                        flux_bc.value_fenics
                        * flux_bc.species.test_function
                        * self.ds(flux_bc.subdomain.id)
                    )

            if isinstance(bc, boundary_conditions.FixedConcentrationBC):
                if bc.enforce_weakly:
                    u = bc.species.solution
                    v = bc.species.test_function
                    # D is set by the material of the volume the surface belongs to
                    vol = self.volume_subdomain_of_surface(bc.subdomain)
                    D = vol.material.get_diffusion_coefficient(
                        self.mesh.mesh, self.temperature_fenics, bc.species
                    )
                    self.formulation += bc.weak_formulation(u, v, self.ds, D)

        for adv_term in self.advection_terms:
            # create vector functionspace based on the elements in the mesh

            for species in adv_term.species:
                conc = species.solution
                v = species.test_function
                vel = adv_term.velocity.fenics_object

                advection_term = ufl.inner(ufl.dot(ufl.grad(conc), vel), v) * self.dx(
                    adv_term.subdomain.id
                )
                self.formulation += advection_term

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
                                if vol in not_defined_in_volume:
                                    not_defined_in_volume.remove(vol)

                    # add c = 0 to formulation where needed
                    for vol in not_defined_in_volume:
                        self.formulation += (
                            spe.solution * spe.test_function * self.dx(vol.id)
                        )

    def update_time_dependent_values(self):
        super().update_time_dependent_values()

        t = float(self.t)

        for reaction in self.reactions:
            for spe in reaction.species:
                if isinstance(spe, _species.ImplicitSpecies):
                    spe.update_density(t=t)
            # the sources built from a reaction only wrap a ufl expression
            # referencing these rate Values, so the rates must be updated here
            # directly. Temperature-dependent rates are ufl expressions
            # referencing temperature_fenics (updated in place below), so only
            # explicitly time-dependent rates need a value update.
            for rate in reaction.rate_coefficients:
                if rate.explicit_time_dependent:
                    rate.update(t=t)

        if (
            isinstance(self.temperature, fem.Function)
            or self.temperature_time_dependent
        ):
            for bc in self.boundary_conditions:
                if isinstance(
                    bc,
                    boundary_conditions.FixedConcentrationBC
                    | boundary_conditions.ParticleFluxBC,
                ):
                    if bc.temperature_dependent:
                        bc.update(t=t)

            for source in self.sources:
                if source.value.temperature_dependent:
                    source.value.update(t=t)

        if self.temperature_time_dependent:
            if isinstance(self.temperature_fenics, fem.Constant):
                self.temperature_fenics.value = self.temperature(t=t)
            elif isinstance(self.temperature_fenics, fem.Function):
                self.temperature_fenics.interpolate(self.temperature_expr)

        for advec_term in self.advection_terms:
            if advec_term.velocity.explicit_time_dependent:
                advec_term.velocity.update(t=t)

    def update_post_processing_solutions(self):
        """Updates the post-processing solutions of each species."""

        for spe in self.species:
            spe.post_processing_solution.x.array[:] = self.u.x.array[
                spe.map_sub_to_main_solution
            ]

    def post_processing(self):
        """Post processes the model."""

        self.update_post_processing_solutions()

        if self.temperature_time_dependent:
            # update global D if temperature time dependent or internal
            # variables time dependent
            # TODO: honestly, we probably don't need to do this at all
            # SurfaceFlux quantities should use ufl.Expr for D instead of a fem.Function

            for spe, D_global in self._species_to_D_global.items():
                D_global.interpolate(self._species_to_D_global_expr[spe])

        for export in self.exports:
            # skip if it isn't time to export
            if hasattr(export, "times"):
                if not is_it_time_to_export(
                    current_time=float(self.t), times=export.times
                ):
                    continue

            # handle VTX exports
            if isinstance(export, exports.ExportBaseClass):
                if isinstance(export, exports.VTXSpeciesExport):
                    if export._checkpoint:
                        for field in export.field:
                            io4dolfinx.write_function(
                                filename=export.filename,
                                u=field.post_processing_solution,
                                time=float(self.t),
                                name=field.name,
                            )
                    else:
                        export.writer.write(float(self.t))
                elif (
                    isinstance(export, exports.VTXTemperatureExport)
                    and self.temperature_time_dependent
                ):
                    self._temperature_as_function.interpolate(
                        self._get_temperature_field_as_function()
                    )
                    export.writer.write(float(self.t))
                elif isinstance(export, exports.CustomFieldExport):
                    # update internal function
                    export.function.interpolate(export.dolfinx_expression)
                    export.writer.write(float(self.t))

            # TODO if export type derived quantity
            if isinstance(export, exports.SurfaceQuantity):
                if isinstance(
                    export,
                    exports.SurfaceFlux | exports.TotalSurface | exports.AverageSurface,
                ):
                    if len(self.advection_terms) > 0:
                        warnings.warn(
                            "Advection terms are not currently accounted for in the "
                            "evaluation of surface flux values"
                        )
                    export.compute(export.field.solution, self.ds)
                else:
                    export.compute()
                # update export data
                export.t.append(float(self.t))

                # if filename given write export data to file
                if export.filename is not None:
                    export.write(t=float(self.t))
            elif isinstance(export, exports.VolumeQuantity):
                if isinstance(export, exports.TotalVolume | exports.AverageVolume):
                    export.compute(u=export.field.solution, dx=self.dx)
                else:
                    export.compute()
                # update export data
                export.t.append(float(self.t))

                # if filename given write export data to file
                if export.filename is not None:
                    export.write(t=float(self.t))
            elif isinstance(export, exports.CustomQuantity):
                is_surface = isinstance(export.subdomain, _subdomain.SurfaceSubdomain)
                measure = self.ds if is_surface else self.dx
                export.compute(measure)

                # update export data
                export.t.append(float(self.t))

                # if filename given write export data to file
                if export.filename is not None:
                    export.write(t=float(self.t))

            if isinstance(export, exports.XDMFExport):
                export.write(float(self.t))

            if isinstance(export, exports.Profile1DExport):
                # computing dofs at each time step is costly so storing it in the export
                if export._dofs is None:
                    index = self.species.index(export.field)
                    V0, export._dofs = self.u.function_space.sub(index).collapse()
                    # dolfinx >=0.11 returns the collapse dof map as a list of
                    # arrays; flatten it back to a 1D index array
                    if Version(dolfinx.__version__) >= Version("0.11"):
                        export._dofs = np.concatenate(export._dofs)
                    coords = V0.tabulate_dof_coordinates()[:, 0]
                    export._sort_coords = np.argsort(coords)
                    x = coords[export._sort_coords]

                    export.x = x

                c = self.u.x.array[export._dofs][export._sort_coords]

                export.data.append(c)
                export.t.append(float(self.t))


class HydrogenTransportProblemDiscontinuous(HydrogenTransportProblem):
    interfaces: list[_subdomain.Interface]
    surface_to_volume: dict
    _method_interface: _subdomain.interface.InterfaceMethod = (
        _subdomain.interface.InterfaceMethod.penalty
    )
    subdomain_to_species: dict

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
        interfaces: list[_subdomain.Interface] | None = None,
        enclosures: list[_Enclosure] | None = None,
        petsc_options: dict | None = None,
    ):
        """Class for a multi-material hydrogen transport problem For other arguments see
        ``festim.HydrogenTransportProblem``.

        Args:
            interfaces (list, optional): list of interfaces (``festim.Interface``
                objects). Defaults to None.
            enclosures (list, optional): list of gas enclosures (``festim.Enclosure``
                objects). Requires dolfinx >= 0.11. Defaults to None.
            surface_to_volume (dict, optional): correspondance dictionary linking
                each ``festim.SurfaceSubdomain`` objects to a ``festim.VolumeSubdomain``
                object). Defaults to None.
            petsc_options (dict, optional): petsc options to be passed to the
                ``festim.NewtonSolver`` object. If None, the default options are:
                ```
                default_petsc_options = {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                }
                ```
                Defaults to None.
        """
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
            petsc_options=petsc_options,
        )
        self.interfaces = interfaces or []
        self.enclosures = enclosures or []
        self.surface_to_volume = {}
        self.subdomain_to_species = {}  # maps subdomain to species defined in it
        self.subdomain_to_V_CG1 = {}
        self._total_volume = None

    @property
    def enclosures(self) -> list[_Enclosure]:
        return self._enclosures

    @enclosures.setter
    def enclosures(self, value):
        value = value or []
        if value:
            # fail early and with a clear message rather than deep inside the
            # function space creation
            check_dolfinx_version_for_enclosures()
        self._enclosures = value

    @property
    def gas_species(self):
        """All the gas species across all the enclosures. Defines the ordering of the
        pressure blocks in the solver."""
        return [gs for enclosure in self.enclosures for gs in enclosure.species]

    @property
    def method_interface(self):
        # deprecation warning
        warnings.warn(
            "The method_interface attribute of the Problem class is deprecated, "
            "please use the method_interface attribute of each interface instead",
            DeprecationWarning,
        )
        return self._method_interface

    @method_interface.setter
    def method_interface(self, value):
        if isinstance(value, _subdomain.interface.InterfaceMethod):
            self._method_interface = value
        elif isinstance(value, str):
            self._method_interface = _subdomain.interface.InterfaceMethod.from_string(
                value
            )
        else:
            raise TypeError("method_interface must be of type str or InterfaceMethod")

    def initialise(self):
        # if method_interface is given as an attribute of Problem class, then pass it to
        # each interface and raise a deprecation warning
        if hasattr(self, "method_interface"):
            warnings.warn(
                "The method_interface attribute of the Problem class is deprecated, "
                "please set the method_interface attribute of each interface instead",
                DeprecationWarning,
            )
            for interface in self.interfaces:
                interface.method = self.method_interface

        # check that all species have a list of F.VolumeSubdomain as this is
        # different from F.HydrogenTransportProblem
        for spe in self.species:
            if not spe.subdomains:
                raise ValueError(
                    f"Species {spe.name} must have a list of subdomains defined "
                    "in 'subdomains' attribute for discontinuous problem"
                )
            if not isinstance(spe.subdomains, list):
                raise TypeError("subdomains attribute in Species should be list")

        self.define_meshtags_and_measures()
        if self.surface_to_volume:
            # tell users that this is no longer required
            warnings.warn(
                f"The surface_to_volume attribute of the {self.__class__.__name__}"
                " class is no longer required and can be removed."
                "The mapping between surface and volume subdomains is now done"
                "automatically based on the connectivity of the mesh and the meshtags",
                DeprecationWarning,
            )
        else:
            facet_to_cell = self.mesh.mesh.topology.connectivity(
                self.mesh.mesh.topology.dim - 1, self.mesh.mesh.topology.dim
            )
            self.surface_to_volume = _subdomain.map_surface_to_volume_subdomains(
                ft=self.facet_meshtags,
                ct=self.volume_meshtags,
                facet_to_cell=facet_to_cell,
                volume_subdomains=self.volume_subdomains,
                surface_subdomains=self.surface_subdomains,
                comm=self.mesh.mesh.comm,
            )

        # create submeshes and transfer meshtags to subdomains
        for subdomain in self.volume_subdomains:
            subdomain.create_subdomain(self.mesh.mesh, self.volume_meshtags)
            subdomain.transfer_meshtag(self.mesh.mesh, self.facet_meshtags)

        for interface in self.interfaces:
            interface.mt = self.volume_meshtags
            interface.parent_mesh = self.mesh.mesh

        self.create_species_from_traps()
        self.link_enclosures()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self._dt = as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )

        self.create_implicit_species_value_fenics()

        for subdomain in self.volume_subdomains:
            self.define_function_spaces(subdomain)

        self.define_enclosure_function_spaces()

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

        self.define_temperature()
        self.convert_enclosure_input_values_to_fenics_objects()
        self.convert_reaction_rates_to_fenics_objects()
        self.create_sources_from_reactions()
        self.convert_source_input_values_to_fenics_objects()
        self.convert_advection_term_to_fenics_objects()
        self.define_boundary_conditions()
        self.create_flux_values_fenics()
        self.create_initial_conditions()

        for subdomain in self.volume_subdomains:
            self.create_subdomain_formulation(subdomain)
            subdomain.u.name = f"u_{subdomain.id}"

        for gas_species in self.gas_species:
            self.create_enclosure_formulation(gas_species)

        self.create_formulation()
        self.create_solver()
        self.initialise_exports()

    def define_temperature(self):
        super().define_temperature()

        # NOTE this won't be needed anymore when https://github.com/FEniCS/dolfinx/pull/4140
        # is released

        # because dolfinx.fem.Expressions cannot work with submeshes
        # (ie. mixing parent and submesh),
        # we need to create "sub" temperature functions for each subdomain
        # pass temperature function to each subdomain
        if isinstance(self.temperature_fenics, fem.Function):
            for subdomain in self.volume_subdomains:
                element_CG = basix.ufl.element(
                    basix.ElementFamily.P,
                    subdomain.submesh.basix_cell(),
                    1,  # could expose?
                    basix.LagrangeVariant.equispaced,
                )
                V = dolfinx.fem.functionspace(subdomain.submesh, element_CG)
                sub_T = dolfinx.fem.Function(V)
                sub_T.name = "temperature"
                from festim.helpers import nmm_interpolate

                nmm_interpolate(f_out=sub_T, f_in=self.temperature_fenics)

                subdomain.sub_T = sub_T

    def create_dirichletbc_form(self, bc: boundary_conditions.FixedConcentrationBC):
        """Creates the ``value_fenics`` attribute for a given
        ``festim.FixedConcentrationBC`` and returns the appropriate
        ``dolfinx.fem.DirichletBC`` object.

        Args:
            bc (festim.FixedConcentrationBC): the dirichlet BC

        Returns:
            dolfinx.fem.DirichletBC: the appropriate dolfinx representation
                generated from ``dolfinx.fem.dirichletbc()``
        """
        fdim = self.mesh.mesh.topology.dim - 1
        volume_subdomain = self.surface_to_volume[bc.subdomain]
        sub_V = bc.species.subdomain_to_function_space[volume_subdomain]
        collapsed_V, _ = sub_V.collapse()

        # in the discontinuous case, if the temperature is given as a function
        # then we can't use the temperature on the parent mesh
        # see issue #1007
        if isinstance(self.temperature_fenics, fem.Function):
            temp = volume_subdomain.sub_T
        else:
            temp = self.temperature_fenics

        bc.create_value(
            temperature=temp,
            function_space=collapsed_V,
            t=self.t,
        )

        volume_subdomain.submesh.topology.create_connectivity(
            volume_subdomain.submesh.topology.dim - 1,
            volume_subdomain.submesh.topology.dim,
        )

        # mapping between sub_function space and collapsed is only needed if
        # value_fenics is a function of the collapsed space
        if isinstance(bc.value_fenics, fem.Function):
            function_space_dofs = (sub_V, collapsed_V)
        else:
            function_space_dofs = sub_V

        bc_dofs = dolfinx.fem.locate_dofs_topological(
            function_space_dofs,
            fdim,
            volume_subdomain.ft.find(bc.subdomain.id),
        )
        form = dolfinx.fem.dirichletbc(bc.value_fenics, bc_dofs, sub_V)
        return form

    def create_initial_conditions(self):
        """For each intial condition, create the value_fenics and assign it to the
        previous solution of the condition's species."""

        for condition in self.initial_conditions:
            idx = self.species.index(condition.species)

            # if the value given is a function, then directly interpolate it on the
            # previous solution of the species
            if isinstance(condition.value, fem.Function):
                nmm_interpolate(condition.volume.u_n.sub(idx), condition.value)

            else:
                V = condition.species.subdomain_to_function_space[condition.volume]

                condition.create_expr_fenics(
                    mesh=self.mesh.mesh,
                    temperature=self.temperature_fenics,
                    function_space=V,
                )

                # assign to previous solution of species
                entities = self.volume_meshtags.find(condition.volume.id)
                condition.volume.u_n.sub(idx).interpolate(
                    condition.expr_fenics, cells1=entities
                )

        for gas_species in self.gas_species:
            gas_species.prev_solution.x.array[:] = gas_species.initial_pressure
            # also seed the current solution: it is the initial guess of the first
            # Newton solve, and some coupling laws (eg. Sieverts' sqrt(P)) are not
            # differentiable at P=0
            gas_species.solution.x.array[:] = gas_species.initial_pressure

    def define_function_spaces(
        self, subdomain: _subdomain.VolumeSubdomain, element_degree=1
    ):
        """Creates appropriate function space and functions for a given subdomain
        (submesh) based on the number of species existing in this subdomain. Then stores
        the functionspace, the current solution (``u``) and the previous solution
        (``u_n``) functions. It also populates the correspondance dicts attributes of
        the species (eg. ``species.subdomain_to_solution``,
        ``species.subdomain_to_test_function``, etc) for easy access to the right
        subfunctions, sub-testfunctions etc.

        Args:
            subdomain (F.VolumeSubdomain): a subdomain of the geometry
            element_degree (int, optional): Degree order for finite element.
                Defaults to 1.
        """
        # get number of species defined in the subdomain
        self.subdomain_to_species[subdomain] = [
            species for species in self.species if subdomain in species.subdomains
        ]

        # instead of using the set function we use a list to keep the order
        unique_species = []
        for species in self.subdomain_to_species[subdomain]:
            if species not in unique_species:
                unique_species.append(species)
        nb_species = len(unique_species)

        element_CG = basix.ufl.element(
            basix.ElementFamily.P,
            subdomain.submesh.basix_cell(),
            element_degree,
            basix.LagrangeVariant.equispaced,
        )
        element = basix.ufl.mixed_element([element_CG] * nb_species)
        V = dolfinx.fem.functionspace(subdomain.submesh, element)
        u = dolfinx.fem.Function(V)
        u_n = dolfinx.fem.Function(V)

        self.subdomain_to_V_CG1[subdomain] = dolfinx.fem.functionspace(
            subdomain.submesh, ("CG", 1)
        )

        # store attributes in the subdomain object
        subdomain.u = u
        subdomain.u_n = u_n

        # split the functions and assign the subfunctions to the species
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
            name = f"{species.name}_{subdomain.id}"
            species.subdomain_to_post_processing_solution[subdomain].name = name

    def convert_source_input_values_to_fenics_objects(self):
        """For each source create the value_fenics."""
        for source in self._unpacked_sources:
            # create value_fenics for all F.ParticleSource objects
            if isinstance(source, _source.ParticleSource):
                V = source.species.subdomain_to_function_space[source.volume]

                source.value.convert_input_value(
                    function_space=V,
                    t=self.t,
                    temperature=self.temperature_fenics,
                    up_to_ufl_expr=True,
                    subdomain=source.volume,
                )

    def convert_advection_term_to_fenics_objects(self):
        """For each advection term convert the input value."""

        for advec_term in self.advection_terms:
            if isinstance(advec_term, AdvectionTerm):
                for spe in advec_term.species:
                    V = spe.subdomain_to_function_space[advec_term.subdomain]
                    advec_term.velocity.convert_input_value(function_space=V, t=self.t)

    def define_boundary_conditions(self):
        for bc in self._unpacked_bcs:
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                bc._volume_subdomain = self.surface_to_volume[bc.subdomain]

        super().define_boundary_conditions()

    def create_dirichletbc_value_ufl(self, bc):
        # as in create_dirichletbc_form, a temperature given as a function lives on the
        # submesh of the volume the surface belongs to (see issue #1007)
        volume_subdomain = self.surface_to_volume[bc.subdomain]
        if isinstance(self.temperature_fenics, fem.Function):
            temperature = volume_subdomain.sub_T
        else:
            temperature = self.temperature_fenics

        bc.create_value_ufl(temperature=temperature)

    def create_subdomain_formulation(self, subdomain: _subdomain.VolumeSubdomain):
        """Creates the variational formulation for each subdomain and stores it in
        ``subdomain.F``

        Args:
            subdomain (F.VolumeSubdomain): a subdomain of the geometry
        """
        form = 0
        # add diffusion and time derivative for each species
        for spe in self.species:
            if subdomain not in spe.subdomains:
                continue
            u = spe.subdomain_to_solution[subdomain]
            u_n = spe.subdomain_to_prev_solution[subdomain]
            v = spe.subdomain_to_test_function[subdomain]

            if self.settings.transient:
                form += ((u - u_n) / self.dt) * v * self.dx(subdomain.id)

            if spe.mobile:
                D = subdomain.material.get_diffusion_coefficient(
                    self.mesh.mesh, self.temperature_fenics, spe
                )
                match self.mesh.coordinate_system:
                    case CoordinateSystem.CARTESIAN:
                        form += ufl.dot(D * ufl.grad(u), ufl.grad(v)) * self.dx(
                            subdomain.id
                        )
                    case CoordinateSystem.CYLINDRICAL:
                        r = ufl.SpatialCoordinate(self.mesh.mesh)[0]
                        form += (
                            r
                            * ufl.dot(D * ufl.grad(u), ufl.grad(v / r))
                            * self.dx(subdomain.id)
                        )
                    case CoordinateSystem.SPHERICAL:
                        r = ufl.SpatialCoordinate(self.mesh.mesh)[0]
                        form += (
                            r**2
                            * ufl.dot(D * ufl.grad(u), ufl.grad(v / r**2))
                            * self.dx(subdomain.id)
                        )
                    case _:
                        raise ValueError(
                            f"Unsupported coordinate system {self.mesh.coordinate_system}"  # noqa: E501
                        )

        # reactions are expanded into particle sources (_unpacked_sources, see
        # create_sources_from_reactions), so they are handled by the source loop

        # add fluxes
        for bc in self._unpacked_bcs:
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                # check that the bc is applied on a surface
                # belonging to this subdomain
                if subdomain == self.surface_to_volume[bc.subdomain]:
                    v = bc.species.subdomain_to_test_function[subdomain]
                    form -= bc.value_fenics * v * self.ds(bc.subdomain.id)
            if isinstance(bc, boundary_conditions.FixedConcentrationBC):
                # as for fluxes, only the subdomain owning the surface gets the term,
                # and its material is the one setting D there
                if (
                    bc.enforce_weakly
                    and subdomain == self.surface_to_volume[bc.subdomain]
                ):
                    u = bc.species.subdomain_to_solution[subdomain]
                    v = bc.species.subdomain_to_test_function[subdomain]
                    D = subdomain.material.get_diffusion_coefficient(
                        self.mesh.mesh, self.temperature_fenics, bc.species
                    )
                    form += bc.weak_formulation(u, v, self.ds, D)

        # add volumetric sources
        for source in self._unpacked_sources:
            if source.volume == subdomain:
                v = source.species.subdomain_to_test_function[subdomain]
                form -= source.value.fenics_object * v * self.dx(subdomain.id)

        # add advection
        for adv_term in self.advection_terms:
            if adv_term.subdomain != subdomain:
                continue

            for spe in adv_term.species:
                v = spe.subdomain_to_test_function[subdomain]
                conc = spe.subdomain_to_solution[subdomain]

                vel = adv_term.velocity.fenics_object
                # print(vel.x.array)
                form += ufl.inner(ufl.dot(ufl.grad(conc), vel), v) * self.dx(
                    subdomain.id
                )

        # store the form in the subdomain object
        subdomain.F = form

    def link_enclosures(self):
        """Validates the enclosures and resolves the links between them, their surfaces
        and the boundary conditions coupled to them.

        This runs before any fenics object is created.
        """
        if not self.enclosures:
            return

        if self.mesh.coordinate_system != CoordinateSystem.CARTESIAN:
            raise NotImplementedError(
                "Enclosures are only supported for cartesian coordinate systems, not "
                f"{self.mesh.coordinate_system!s}. The surface integrals of the "
                "pressure balance would need the appropriate metric factors."
            )

        mesh_dim = self.mesh.mesh.topology.dim
        all_gas_species = self.gas_species

        # A GasSpecies carries exactly one pressure unknown and one backref to its
        # enclosure, so it cannot belong to two of them. Sharing one silently corrupts
        # the model (the backref points to the last enclosure, and the species gets two
        # pressure blocks sharing one solution). Catch it explicitly.
        seen = set()
        for gas_species in all_gas_species:
            if id(gas_species) in seen:
                enclosures = [e for e in self.enclosures if gas_species in e.species]
                names = ", ".join(str(e) for e in enclosures)
                raise ValueError(
                    f"{gas_species} belongs to more than one enclosure ({names}). A "
                    "GasSpecies has a single partial pressure and can live in only one "
                    "enclosure; create a separate GasSpecies for each enclosure."
                )
            seen.add(id(gas_species))

        for enclosure in self.enclosures:
            for surface in enclosure.surfaces:
                if surface not in self.surface_subdomains:
                    raise ValueError(
                        f"Surface {surface.id} of {enclosure} is not in the subdomains "
                        "of the model"
                    )
            # The flux through a surface is per unit area, so turning it into a number
            # of particles per second needs the physical area of that surface. Only a 3D
            # mesh measures that area itself: in 1D a surface is a point and in 2D a
            # line, so the missing extent has to come from the user.
            if enclosure.surfaces and not enclosure.areas_given and mesh_dim < 3:
                missing = (
                    "area (m2) of the surface"
                    if mesh_dim == 1
                    else ("out-of-plane depth (m) of the model")
                )
                raise ValueError(
                    f"{enclosure} is attached to surfaces on a {mesh_dim}D mesh, so "
                    "the areas of those surfaces cannot be taken from the mesh and "
                    "must be given: pass surfaces as a dict mapping each surface to "
                    f"the {missing}, eg. surfaces={{my_surface: 1e-4}}."
                )
            # a connection only needs declaring on one side: mirror it onto the partner
            for opening in enclosure.openings:
                if not isinstance(opening, EnclosureConnection):
                    continue
                for gas_species in opening.species:
                    if gas_species not in all_gas_species:
                        raise ValueError(
                            f"{gas_species} is connected by an EnclosureConnection but "
                            "does not belong to any enclosure of the model"
                        )
                    partner = gas_species.enclosure
                    if opening not in partner.openings:
                        partner.openings.append(opening)

        # let the boundary conditions know which gas species they are coupled to
        for bc in self._unpacked_bcs:
            pressure = getattr(bc, "gas_pressure", None)
            if pressure is None:
                pressure = getattr(bc, "pressure", None)
            if isinstance(pressure, _GasSpecies):
                if pressure not in all_gas_species:
                    raise ValueError(
                        f"{bc} is coupled to {pressure}, which does not belong to any "
                        "enclosure of the model"
                    )
                bc._gas_species = pressure

                if isinstance(bc, boundary_conditions.FixedConcentrationBC):
                    # A Dirichlet value that depends on the pressure cannot be
                    # interpolated into a fem.Function, so it can only be enforced
                    # weakly. Enabling that silently would change the discretisation
                    # behind the user's back, and there is no defensible default
                    # penalty, so ask for both explicitly.
                    if not bc.enforce_weakly or bc.penalty is None:
                        raise ValueError(
                            f"{type(bc).__name__} on surface {bc.subdomain.id} is "
                            f"coupled to {pressure}, whose pressure is an unknown of "
                            "the problem. Such a boundary condition can only be "
                            "enforced weakly: pass enforce_weakly=True and a penalty "
                            "(a dimensionless value of order 10-100)."
                        )

    def define_enclosure_function_spaces(self):
        """Creates a real function space, a solution and a previous solution for each
        gas species of each enclosure.

        The real function spaces live on the parent mesh: every form of the blocked
        system is integrated over the parent mesh, with submesh functions pulled in via
        ``entity_maps``.
        """
        if not self.enclosures:
            return

        mesh = self.mesh.mesh
        # The real space is defined on the parent mesh. It could instead live on a
        # submesh of the contact facets (as in the scifem demo), but that is a
        # code-structure choice, not a performance one: a real space has a single
        # global dof either way, and the pressure's Jacobian coupling is already
        # surface-local (its sparsity comes from the ds integration measure, not from
        # the mesh the space is defined on). A submesh would only add entity_map
        # bookkeeping.
        for gas_species in self.gas_species:
            V = create_real_function_space(mesh)
            gas_species.function_space = V
            gas_species.solution = fem.Function(V)
            gas_species.prev_solution = fem.Function(V)
            gas_species.test_function = ufl.TestFunction(V)
            gas_species.solution.name = f"P_{gas_species.name}"

        # The pressure balance is a 0D equation, but its test function is a single
        # global constant, so every term of it must be written as an integral. The
        # scalar terms are therefore spread over a region and divided by the measure of
        # that region, which recovers the bare scalar: for constant f,
        # int_R f/|R| q dR == f. The region is the enclosure's contact surfaces where it
        # has some (that is where the physics is, and it keeps assembly on the boundary
        # facets), and the whole domain for an enclosure that only has openings.
        one = fem.Constant(mesh, 1.0)
        self._total_volume = scifem.assemble_scalar(fem.form(one * self.dx))
        for enclosure in self.enclosures:
            enclosure._contact_measure = sum(
                scifem.assemble_scalar(fem.form(one * self.ds(surface.id)))
                for surface in enclosure.surfaces
            )

    def convert_enclosure_input_values_to_fenics_objects(self):
        """Converts the user input values of the enclosures and their openings to fenics
        objects."""
        for enclosure in self.enclosures:
            # opening parameters are scalars, so any function space on the parent mesh
            # will do here
            enclosure.convert_input_values_to_fenics_objects(
                function_space=self.V_DG_0, t=self.t
            )

    # NOTE: what are the alternative naming for "production rate"?
    def gas_production_rates(self, surface, gas_species: _GasSpecies):
        """The rates at which particles of a gas species are produced at a surface, in
        particles/s/m2, positive when entering the gas.

        Args:
            surface: the surface subdomain
            gas_species: the gas species

        Yields:
            ufl expressions for each contribution at that surface
        """
        for bc in self.boundary_conditions:
            if bc.subdomain is not surface:
                continue
            if (
                isinstance(bc, boundary_conditions.SurfaceReactionBC)
                and bc.gas_pressure is gas_species
            ):
                # value_fenics is the rate at which the solid gains particles, so the
                # gas loses them. Taken once per reaction, not once per reactant: the
                # partials share the same expression and differ only in which species
                # of the solid they apply to.
                yield -bc.flux_bcs[0].value_fenics

            if (
                isinstance(bc, boundary_conditions.FixedConcentrationBC)
                and getattr(bc, "_gas_species", None) is gas_species
            ):
                # A weakly enforced Dirichlet BC only satisfies u = value up to O(h^p),
                # so the raw -D grad(u).n is not what the discrete scheme conserves.
                # Using the numerical flux instead makes what the solid loses equal what
                # the gas gains exactly, rather than to within the Nitsche consistency
                # error, which would otherwise accumulate in the pressure.
                volume_subdomain = self.surface_to_volume[surface]
                u = bc.species.subdomain_to_solution[volume_subdomain]
                D = volume_subdomain.material.get_diffusion_coefficient(
                    self.mesh.mesh, self.temperature_fenics, bc.species
                )
                # the flux is in particles of the solid species per second per m2;
                # the stoichiometry converts it to molecules of the gas species
                yield bc.numerical_flux(u, D, self.mesh.mesh) / bc.stoichiometry

    def create_enclosure_formulation(self, gas_species: _GasSpecies):
        """Creates the variational formulation of the pressure balance of a gas species
        and stores it in ``gas_species.F``.

        Args:
            gas_species: the gas species
        """
        enclosure = gas_species.enclosure
        P = gas_species.solution
        P_n = gas_species.prev_solution
        q = gas_species.test_function
        kT = enclosure.thermal_energy

        # see define_enclosure_function_spaces: a scalar term of this 0D equation is
        # spread over a region and divided by the measure of that region
        if enclosure.surfaces:
            regions = [self.ds(surface.id) for surface in enclosure.surfaces]
            measure_of_regions = enclosure._contact_measure
        else:
            regions = [self.dx]
            measure_of_regions = self._total_volume

        def as_integral(scalar):
            return sum(scalar / measure_of_regions * q * region for region in regions)

        form = 0

        if self.settings.transient:
            form += as_integral((P - P_n) / self.dt)

        for surface, area in enclosure.surfaces.items():
            for rate in self.gas_production_rates(surface, gas_species):
                # rate is per unit area, so the physical area of the surface turns it
                # into particles per second. This integral is already over the surface
                # and must not be normalised.
                form -= kT / enclosure.volume * area * rate * q * self.ds(surface.id)

        for opening in enclosure.openings:
            if not opening.applies_to(gas_species):
                continue
            flow_rate = opening.molar_flow_rate(gas_species, enclosure)
            form -= as_integral(kT / enclosure.volume * flow_rate)

        # The pressure is only determined if it appears in its own balance. In a
        # transient problem the time derivative always puts it there. In steady state it
        # only appears through something that depends on it: a surface reaction coupled
        # to this species, or a pressure-dependent opening (Pump, Reservoir,
        # EnclosureConnection). A PrescribedFlowRate alone does not, and neither does an
        # enclosure with no coupling at all.
        determined = isinstance(
            form, ufl.Form
        ) and P in ufl.algorithms.extract_coefficients(form)
        if not determined:
            raise ValueError(
                f"The pressure of {gas_species} in {enclosure} is undetermined: it "
                "does not appear in its own mass balance, so the problem has no unique "
                "solution. This happens in steady state when nothing in the balance "
                "depends on the pressure. Give the enclosure a surface with a "
                "SurfaceReactionBC coupled to this species, or a pressure-dependent "
                "opening (Pump, Reservoir or EnclosureConnection), or run a transient "
                "simulation."
            )

        gas_species.F = form

    def create_formulation(self):
        """Takes all the formulations for each subdomain and adds the interface
        conditions.

        Finally compute the jacobian matrix and store it in the ``J`` attribute,
        adds the ``entity_maps`` to the forms and store them in the ``forms`` attribute
        """
        mesh = self.mesh.mesh
        mt = self.facet_meshtags

        for interface in self.interfaces:
            interface.mesh = mesh
            interface.mt = mt

        integral_data = [
            interface.compute_mapped_interior_facet_data(mesh)
            for interface in self.interfaces
        ]
        dInterface = ufl.Measure("dS", domain=mesh, subdomain_data=integral_data)

        all_mobile_species = [spe for spe in self.species if spe.mobile]
        for interface in self.interfaces:
            F_0, F_1 = interface.get_formulation(
                dInterface,
                species=all_mobile_species,
                temperature=self.temperature_fenics,
            )
            subdomain_0, subdomain_1 = interface.subdomains
            subdomain_0.F += F_0
            subdomain_1.F += F_1

        # the unknowns of the blocked system: one block per volume subdomain, plus one
        # block per gas species (the pressure of that species in its enclosure)
        all_forms = [subdomain.F for subdomain in self.volume_subdomains] + [
            gas_species.F for gas_species in self.gas_species
        ]
        all_unknowns = [subdomain.u for subdomain in self.volume_subdomains] + [
            gas_species.solution for gas_species in self.gas_species
        ]

        J = []
        # this is the symbolic differentiation of the Jacobian
        for form in all_forms:
            jac = []
            for unknown in all_unknowns:
                jac.append(
                    ufl.derivative(form, unknown),
                )
            J.append(jac)
        # compile jacobian (J) and residual (F)
        entity_maps = [sd.cell_map for sd in self.volume_subdomains]

        self.forms = dolfinx.fem.form(
            all_forms,
            entity_maps=entity_maps,
            jit_options={
                "cffi_extra_compile_args": ["-O3", "-march=native"],
                "cffi_libraries": ["m"],
            },
        )
        self.J = dolfinx.fem.form(
            J,
            entity_maps=entity_maps,
            jit_options={
                "cffi_extra_compile_args": ["-O3", "-march=native"],
                "cffi_libraries": ["m"],
            },
        )

    def create_solver(self):

        petsc_options = self.get_petsc_options()

        self.solver = NonlinearProblem(
            self.forms,
            [subdomain.u for subdomain in self.volume_subdomains]
            + [gas_species.solution for gas_species in self.gas_species],
            bcs=self.bc_forms,
            J=self.J,
            petsc_options=petsc_options,
            petsc_options_prefix="festim_solver",
            kind="mpi",
        )

        self.solver.solver.setMonitor(SnesMonitor)
        self.solver.solver.getKSP().setMonitor(KSPMonitor)
        self.solver.solver.setConvergenceTest(convergenceTest)

        # Delete PETSc options post setting them, ref:
        # https://gitlab.com/petsc/petsc/-/issues/1201
        snes = self.solver.solver
        prefix = snes.getOptionsPrefix()
        opts = PETSc.Options()
        for k in petsc_options.keys():
            del opts[f"{prefix}{k}"]

    def create_flux_values_fenics(self):
        """For each particle flux create the ``value_fenics`` attribute."""
        for bc in self._unpacked_bcs:
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                volume_subdomain = self.surface_to_volume[bc.subdomain]
                bc.create_value_fenics(
                    mesh=volume_subdomain.submesh,
                    temperature=self.temperature_fenics,
                    t=self.t,
                )

    def initialise_exports(self):
        for export in self.exports:
            if isinstance(export, exports.VTXSpeciesExport):
                functions = export.get_functions()
                if not export._checkpoint:
                    export.writer = dolfinx.io.VTXWriter(
                        functions[0].function_space.mesh.comm,
                        export.filename,
                        functions,
                        engine="BP5",
                    )
                else:
                    io4dolfinx.write_mesh(
                        filename=export.filename,
                        mesh=functions[0].function_space.mesh,
                        backend="adios2",
                    )
            elif isinstance(export, exports.VTXTemperatureExport):
                assert isinstance(self.temperature_fenics, fem.Function), (
                    "Temperature must be space-dependent to be exported as "
                    "VTXTemperatureExport"
                )
                export.writer = dolfinx.io.VTXWriter(
                    self.temperature_fenics.function_space.mesh.comm,
                    export.filename,
                    self.temperature_fenics,
                    engine="BP5",
                )
            elif isinstance(export, exports.CustomFieldExport):
                # need to find an appropriate function space on the right submesh
                V = self.subdomain_to_V_CG1[export.subdomain]
                export.function = fem.Function(V)
                export.set_dolfinx_expression(
                    # need to pass the right temperature
                    temperature=self.temperature_fenics,
                    time=self.t,
                )

                export.writer = dolfinx.io.VTXWriter(
                    comm=export.function.function_space.mesh.comm,
                    filename=export.filename,
                    output=export.function,
                    engine="BP5",
                )

        # compute diffusivity function for surface fluxes
        # for the discontinuous case, we don't use D_global as in
        # HydrogenTransportProblem
        for export in self.exports:
            if isinstance(export, exports.SurfaceQuantity):
                volume = self.surface_to_volume[export.surface]
                D = volume.material.get_diffusion_coefficient(
                    self.mesh.mesh, self.temperature_fenics, export.field
                )
                # NOTE: maybe we need to make sure there are no functionspace clashes?

                export.D = D

            # reset the data and time for SurfaceQuantity and VolumeQuantity
            if isinstance(export, exports.DerivedQuantity):
                export.t = []
                export.data = []

            if isinstance(export, exports.CustomQuantity):
                volume = (
                    export.subdomain
                    if not isinstance(export.subdomain, _subdomain.SurfaceSubdomain)
                    else self.surface_to_volume[
                        export.subdomain
                        if isinstance(export.subdomain, _subdomain.SurfaceSubdomain)
                        else next(
                            s
                            for s in self.surface_subdomains
                            if s.id == export.subdomain
                        )
                    ]
                )

                kwargs = {
                    species.name: species.subdomain_to_post_processing_solution[volume]
                    for species in self.species
                }
                kwargs["n"] = ufl.FacetNormal(self.mesh.mesh)
                kwargs["t"] = self.t
                kwargs["T"] = self.temperature_fenics

                D_kwargs = {
                    f"D_{sp.name}": volume.material.get_diffusion_coefficient(
                        self.mesh.mesh, self.temperature_fenics, sp
                    )
                    for sp in self.species
                }
                kwargs.update(D_kwargs)
                kwargs["D"] = {sp.name: D_kwargs[f"D_{sp.name}"] for sp in self.species}
                if len(self.species) == 1:
                    kwargs["D"] = kwargs[f"D_{self.species[0].name}"]
                kwargs["x"] = ufl.SpatialCoordinate(self.mesh.mesh)
                export.ufl_expr = export.expr(**kwargs)

    def post_processing(self):
        # update post-processing solutions (for each species in each subdomain)
        # with new solution
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
            # skip if it isn't time to export
            if hasattr(export, "times"):
                if not is_it_time_to_export(
                    current_time=float(self.t), times=export.times
                ):
                    continue
            # handle VTX exports
            if isinstance(export, exports.ExportBaseClass):
                if isinstance(export, exports.CustomFieldExport):
                    # update internal function
                    export.function.interpolate(export.dolfinx_expression)
                    export.writer.write(float(self.t))
                elif isinstance(export, exports.VTXSpeciesExport):
                    if export._checkpoint:
                        for species in export.field:
                            post_processing_solution = (
                                species.subdomain_to_post_processing_solution[
                                    export._subdomain
                                ]
                            )
                            io4dolfinx.write_function(
                                filename=export.filename,
                                u=post_processing_solution,
                                time=float(self.t),
                                name=species.name,
                            )
                    else:
                        export.writer.write(float(self.t))
                elif isinstance(export, exports.VTXTemperatureExport):
                    export.writer.write(float(self.t))
                else:
                    raise NotImplementedError(
                        f"Export type {type(export)} not implemented"
                    )
            # handle derived quantities
            if isinstance(export, exports.SurfaceQuantity):
                if isinstance(
                    export,
                    exports.SurfaceFlux | exports.TotalSurface | exports.AverageSurface,
                ):
                    if len(self.advection_terms) > 0:
                        warnings.warn(
                            "Advection terms are not currently accounted for in the "
                            "evaluation of surface flux values"
                        )
                    export_surf = export.surface
                    export_vol = self.surface_to_volume[export_surf]
                    submesh_function = (
                        export.field.subdomain_to_post_processing_solution[export_vol]
                    )
                    entity_maps = [sd.cell_map for sd in self.volume_subdomains]

                    export.compute(
                        u=submesh_function, ds=self.ds, entity_maps=entity_maps
                    )
                else:
                    export.compute()

            elif isinstance(export, exports.VolumeQuantity):
                if isinstance(export, exports.TotalVolume | exports.AverageVolume):
                    entity_maps = [sd.cell_map for sd in self.volume_subdomains]

                    export.compute(
                        u=export.field.subdomain_to_post_processing_solution[
                            export.volume
                        ],
                        dx=self.dx,
                        entity_maps=entity_maps,
                    )
                else:
                    export.compute()

            elif isinstance(export, exports.CustomQuantity):
                is_surface = isinstance(export.subdomain, _subdomain.SurfaceSubdomain)
                measure = self.ds if is_surface else self.dx

                # getting entity_maps
                entity_maps = [sd.cell_map for sd in self.volume_subdomains]
                export.compute(measure, entity_maps=entity_maps)

            elif isinstance(export, exports.GasPressure):
                export.compute()

            if isinstance(export, exports.DerivedQuantity):
                # update export data
                export.t.append(float(self.t))

                # if filename given write export data to file
                if export.filename is not None:
                    export.write(t=float(self.t))

            elif isinstance(export, exports.Profile1DExport):
                assert export.subdomain, (
                    "Profile1DExport requires a subdomain to be set"
                )
                u = export.subdomain.u
                if export._dofs is None:
                    index = self.subdomain_to_species[export.subdomain].index(
                        export.field
                    )
                    V0, export._dofs = u.function_space.sub(index).collapse()
                    # dolfinx >=0.11 returns the collapse dof map as a list of
                    # arrays; flatten it back to a 1D index array
                    if Version(dolfinx.__version__) >= Version("0.11"):
                        export._dofs = np.concatenate(export._dofs)
                    coords = V0.tabulate_dof_coordinates()[:, 0]
                    export._sort_coords = np.argsort(coords)
                    x = coords[export._sort_coords]
                    export.x = x

                c = u.x.array[export._dofs][export._sort_coords]

                export.data.append(c)
                export.t.append(float(self.t))

    def update_time_dependent_values(self):
        super().update_time_dependent_values()

        for enclosure in self.enclosures:
            enclosure.update_time_dependent_values(t=float(self.t))

        # update sub_T if temperature is given as a function
        if self.temperature_time_dependent:
            if isinstance(self.temperature_fenics, fem.Function):
                for subdomain in self.volume_subdomains:
                    temp = self.temperature_fenics
                    sub_T = subdomain.sub_T
                    from festim.helpers import nmm_interpolate

                    nmm_interpolate(f_out=sub_T, f_in=temp)

    def iterate(self):
        """Iterates the model for a given time step."""
        if self.show_progress_bar:
            self.progress_bar.update(
                min(self.dt.value, abs(self.settings.final_time - self.t.value))
            )
        self.t.value += self.dt.value

        self.update_time_dependent_values()

        # Solve main problem
        _ = self.solver.solve()
        converged_reason = self.solver.solver.getConvergedReason()
        assert converged_reason > 0, (
            f"Non-linear solver did not converge. Reason code: {converged_reason}. \n See https://petsc.org/release/manualpages/SNES/SNESConvergedReason/ for more information."  # noqa: E501
        )
        nb_its = self.solver.solver.getIterationNumber()

        # post processing
        self.post_processing()

        # update previous solution
        for subdomain in self.volume_subdomains:
            subdomain.u_n.x.array[:] = subdomain.u.x.array[:]
        for gas_species in self.gas_species:
            gas_species.prev_solution.x.array[:] = gas_species.solution.x.array[:]

        # adapt stepsize
        if self.settings.stepsize.adaptive:
            new_stepsize = self.settings.stepsize.modify_value(
                value=self.dt.value, nb_iterations=nb_its, t=self.t.value
            )
            self.dt.value = new_stepsize

    def run(self):
        if self.settings.transient:
            # Solve transient
            if self.show_progress_bar:
                self.progress_bar = tqdm.auto.tqdm(
                    desc=f"Solving {self.__class__.__name__}",
                    total=self.settings.final_time,
                    unit_scale=True,
                )
            while self.t.value < self.settings.final_time:
                self.iterate()
            if self.show_progress_bar:
                self.progress_bar.refresh()  # refresh progress bar to show 100%
                self.progress_bar.close()
        else:
            # Solve steady-state
            self.solver.solve()
            self.post_processing()

    def __del__(self):
        for export in self.exports:
            if isinstance(export, exports.ExportBaseClass):
                if hasattr(export, "writer") and export.writer is not None:
                    export.writer.close()


class HydrogenTransportProblemDiscontinuousChangeVar(HydrogenTransportProblem):
    species: list[_species.Species]

    def initialise(self):
        # check if a SurfaceReactionBC is given
        for bc in self.boundary_conditions:
            if isinstance(bc, (boundary_conditions.SurfaceReactionBC)):
                raise ValueError(
                    f"{type(bc)} not implemented for "
                    f"HydrogenTransportProblemDiscontinuousChangeVar"
                )
            # with the change of variable, the solution of a mobile species is the
            # chemical potential and not the concentration, so a species-dependent
            # value would silently be given the wrong quantity
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                if bc.species_dependent_value:
                    raise ValueError(
                        f"{type(bc)} concentration-dependent not implemented for "
                        f"HydrogenTransportProblemDiscontinuousChangeVar"
                    )

        for source in self.sources:
            if isinstance(source, _source.ParticleSource):
                if source.value.species_dependent:
                    raise ValueError(
                        f"{type(source)} concentration-dependent not implemented for "
                        f"HydrogenTransportProblemDiscontinuousChangeVar"
                    )

        super().initialise()

    def create_sources_from_reactions(self):
        # this problem uses a change of variable (u * K_S) for mobile species, so
        # reactions cannot be expressed as plain particle sources and are instead
        # handled directly in add_reaction_term. _unpacked_sources therefore holds
        # only the user sources. This override will be removed when this deprecated
        # problem class is dropped.
        self._unpacked_sources = list(self.sources)

    def create_formulation(self):
        """Creates the formulation of the model."""

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
                    K_S = vol.material.get_solubility_coefficient(
                        self.mesh.mesh, self.temperature_fenics, spe
                    )
                    c = u * K_S
                    c_n = u_n * K_S
                else:
                    c = u
                    c_n = u_n
                if spe.mobile:
                    self.formulation += ufl.dot(D * ufl.grad(c), ufl.grad(v)) * self.dx(
                        vol.id
                    )

                if self.settings.transient:
                    self.formulation += ((c - c_n) / self.dt) * v * self.dx(vol.id)

        for reaction in self.reactions:
            self.add_reaction_term(reaction)

        # add sources (reactions are handled inline above, not expanded into
        # sources, so _unpacked_sources is just the user sources here)
        for source in self._unpacked_sources:
            self.formulation -= (
                source.value.fenics_object
                * source.species.test_function
                * self.dx(source.volume.id)
            )

        # add fluxes
        for bc in self._unpacked_bcs:
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
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
                            if (
                                spe in reaction.product
                            ):  # TODO we probably need this in HydrogenTransportProblem
                                # too no?
                                if vol == reaction.volume:
                                    if vol in not_defined_in_volume:
                                        not_defined_in_volume.remove(vol)

                    # add c = 0 to formulation where needed
                    for vol in not_defined_in_volume:
                        self.formulation += (
                            spe.solution * spe.test_function * self.dx(vol.id)
                        )

    def add_reaction_term(self, reaction: _reaction.GenericReaction):
        """Adds the reaction term to the formulation."""

        products = (
            reaction.product
            if isinstance(reaction.product, list)
            else [reaction.product]
        )

        # we cannot use the `concentration` attribute of the mobile species and need to
        # use u * K_S instead

        def get_concentrations(species_list) -> list:
            concentrations = []
            for spe in species_list:
                if isinstance(spe, _species.ImplicitSpecies):
                    concentrations.append(None)
                elif spe.mobile:
                    K_S = reaction.volume.material.get_solubility_coefficient(
                        self.mesh.mesh, self.temperature_fenics, spe
                    )
                    concentrations.append(spe.solution * K_S)
                else:
                    concentrations.append(None)
            return concentrations

        reactant_concentrations = get_concentrations(reaction.reactant)
        product_concentrations = get_concentrations(products)

        # get the reaction term from the reaction
        reaction_term = reaction.reaction_term(
            reactant_concentrations=reactant_concentrations,
            product_concentrations=product_concentrations,
        )

        # add reaction term to formulation
        # reactant
        for reactant in reaction.reactant:
            if isinstance(reactant, _species.Species):
                self.formulation += (
                    reaction_term * reactant.test_function * self.dx(reaction.volume.id)
                )

        # product
        for product in products:
            self.formulation += (
                -reaction_term * product.test_function * self.dx(reaction.volume.id)
            )

    def initialise_exports(self):
        self.override_post_processing_solution()
        super().initialise_exports()

    def override_post_processing_solution(self):
        # override the post-processing solution c = theta * K_S
        Q0 = fem.functionspace(self.mesh.mesh, ("DG", 0))
        Q1 = fem.functionspace(self.mesh.mesh, ("DG", 1))

        for spe in self.species:
            if not spe.mobile:
                continue
            K_S0 = fem.Function(Q0)
            E_KS = fem.Function(Q0)
            for subdomain in self.volume_subdomains:
                entities = self.volume_meshtags.find(subdomain.id)
                K_S0.x.array[entities] = subdomain.material.get_K_S_0(spe)
                E_KS.x.array[entities] = subdomain.material.get_E_K_S(spe)

            K_S = K_S0 * ufl.exp(-E_KS / (k_B * self.temperature_fenics))

            theta = spe.solution

            spe.dg_expr = fem.Expression(theta * K_S, Q1.element.interpolation_points)
            spe.post_processing_solution = fem.Function(Q1)
            spe.post_processing_solution.interpolate(
                spe.dg_expr
            )  # NOTE: do we need this line since it's in initialise?

    def update_post_processing_solutions(self):
        """Updates the post-processing solutions after each time step."""
        # need to compute c = theta * K_S
        # this expression is stored in species.dg_expr

        for spe in self.species:
            if not spe.mobile:
                continue
            spe.post_processing_solution.interpolate(spe.dg_expr)

    def create_dirichletbc_form(self, bc: boundary_conditions.FixedConcentrationBC):
        """Creates a dirichlet boundary condition form.

        Args:
            bc (festim.DirichletBC): the boundary condition

        Returns:
            dolfinx.fem.bcs.DirichletBC: A representation of
                the boundary condition for modifying linear systems.
        """
        # create K_S function
        Q0 = fem.functionspace(self.mesh.mesh, ("DG", 0))
        K_S0 = fem.Function(Q0)
        E_KS = fem.Function(Q0)
        for subdomain in self.volume_subdomains:
            entities = self.volume_meshtags.find(subdomain.id)
            K_S0.x.array[entities] = subdomain.material.get_K_S_0(bc.species)
            E_KS.x.array[entities] = subdomain.material.get_E_K_S(bc.species)

        K_S = K_S0 * ufl.exp(-E_KS / (k_B * self.temperature_fenics))

        # create value_fenics
        bc.create_value(
            temperature=self.temperature_fenics,
            function_space=bc.species.collapsed_function_space,
            t=self.t,
            K_S=K_S,
        )

        # get dofs
        if isinstance(bc.value_fenics, fem.Function):
            function_space_dofs = (
                bc.species.sub_function_space,
                bc.species.collapsed_function_space,
            )
        else:
            function_space_dofs = bc.species.sub_function_space

        bc_dofs = bc.define_surface_subdomain_dofs(
            facet_meshtags=self.facet_meshtags,
            function_space=function_space_dofs,
        )

        # create form
        form = fem.dirichletbc(
            value=bc.value_fenics,
            dofs=bc_dofs,
            V=bc.species.sub_function_space,
        )

        return form

    def update_time_dependent_values(self):
        super().update_time_dependent_values()

        if self.temperature_time_dependent:
            for bc in self.boundary_conditions:
                if isinstance(bc, boundary_conditions.FixedConcentrationBC):
                    bc.update(self.t)


class HydrogenTransportProblemDG(HydrogenTransportProblem):
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
        petsc_options: dict | None = None,
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
            petsc_options=petsc_options,
        )

    def initialise(self):
        self.create_species_from_traps()
        self.define_function_spaces(element_degree=self.settings.element_degree)
        self.define_meshtags_and_measures()
        self.assign_functions_to_species()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self._dt = as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )

        self.create_implicit_species_value_fenics()

        self.define_temperature()
        self.define_boundary_conditions()
        self.convert_source_input_values_to_fenics_objects()
        self.convert_advection_term_to_fenics_objects()
        self.create_flux_values_fenics()
        self.create_initial_conditions()
        self.create_formulation()
        self.create_solver()
        self.initialise_exports()

    def define_function_spaces(self, element_degree: int = 1):
        """Creates the function space of the modelw with a mixed element. Creates the
        main solution and previous solution function u and u_n. Create global DG
        function spaces of degree 0 and 1 for the global diffusion coefficient.

        Args:
            element_degree: Degree order for finite element. Defaults to 1.
        """

        element_DG = basix.ufl.element(
            "DG",
            self.mesh.mesh.basix_cell(),
            element_degree,
            basix.LagrangeVariant.equispaced,
        )

        elements = []
        for spe in self.species:
            if isinstance(spe, _species.Species):
                elements.append(element_DG)

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

    def create_formulation(self):
        """Creates the formulation of the model."""

        self.formulation = 0

        # add diffusion and time derivative for each species
        for spe in self.species:
            u = spe.solution
            u_n = spe.prev_solution
            v = spe.test_function

            for vol in self.volume_subdomains:
                if spe.mobile:
                    D = vol.material.get_diffusion_coefficient(
                        self.mesh.mesh, self.temperature_fenics, spe
                    )
                    match self.mesh.coordinate_system:
                        case CoordinateSystem.CARTESIAN:
                            self.formulation += ufl.dot(
                                D * ufl.grad(u), ufl.grad(v)
                            ) * self.dx(vol.id)
                        case CoordinateSystem.CYLINDRICAL:
                            r = ufl.SpatialCoordinate(self.mesh.mesh)[0]
                            self.formulation += (
                                r
                                * ufl.dot(D * ufl.grad(u), ufl.grad(v / r))
                                * self.dx(vol.id)
                            )
                        case CoordinateSystem.SPHERICAL:
                            r = ufl.SpatialCoordinate(self.mesh.mesh)[0]
                            self.formulation += (
                                r**2
                                * ufl.dot(D * ufl.grad(u), ufl.grad(v / r**2))
                                * self.dx(vol.id)
                            )
                        case _:
                            raise NotImplementedError(
                                f"Unknown coordinate system {self.mesh.coordinate_system!s}"  # noqa: E501
                            )

                    # Add SIPG diffusion
                    self.formulation += (
                        -D
                        * ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(v, self.mesh.n))
                        * self.dS
                    )
                    self.formulation += (
                        -D
                        * ufl.inner(ufl.jump(u, self.mesh.n), ufl.avg(ufl.grad(v)))
                        * self.dS
                    )
                    PENALY = 100
                    self.formulation += (
                        D
                        * (PENALY / ufl.avg(self.mesh.h))
                        * ufl.inner(ufl.jump(u, self.mesh.n), ufl.jump(v, self.mesh.n))
                        * self.dS
                    )

                if self.settings.transient:
                    self.formulation += ((u - u_n) / self.dt) * v * self.dx(vol.id)

        for reaction in self.reactions:
            for reactant in reaction.reactant:
                if isinstance(reactant, _species.Species):
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
                source.value.fenics_object
                * source.species.test_function
                * self.dx(source.volume.id)
            )

        # add fluxes
        for bc in self.boundary_conditions:
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                self.formulation -= (
                    bc.value_fenics
                    * bc.species.test_function
                    * self.ds(bc.subdomain.id)
                )
            if isinstance(bc, boundary_conditions.SurfaceReactionBC):
                for flux_bc in bc.flux_bcs:
                    self.formulation -= (
                        flux_bc.value_fenics
                        * flux_bc.species.test_function
                        * self.ds(flux_bc.subdomain.id)
                    )
            if isinstance(bc, boundary_conditions.FixedConcentrationBC):
                if not bc.enforce_weakly:
                    raise ValueError(
                        "FixedConcentrationBC must be enforced weakly for DG formulation"
                    )
                u = bc.species.solution
                v = bc.species.test_function
                self.formulation += bc.weak_formulation(u, v, self.ds)

            if isinstance(bc, boundary_conditions.FixedConcentrationInflowBC):
                if not bc.enforce_weakly:
                    raise ValueError(
                        "FixedConcentrationInflowBC must be enforced weakly "
                        "for DG formulation"
                    )
                u = bc.species.solution
                v = bc.species.test_function
                vel = bc.velocity.fenics_object

                lmbda = ufl.conditional(ufl.gt(ufl.dot(vel, self.mesh.n), 0), 1, 0)

                self.formulation += bc.weak_formulation(u, v, self.ds)

                self.formulation += -ufl.inner(
                    (1 - lmbda) * ufl.dot(vel, self.mesh.n) * u, v
                ) * self.ds(bc.subdomain.id)

            if isinstance(bc, boundary_conditions.OutflowBC):
                for species in bc.species:
                    conc = species.solution
                    v = species.test_function
                    vel = bc.velocity.fenics_object

                    lmbda = ufl.conditional(ufl.gt(ufl.dot(vel, self.mesh.n), 0), 1, 0)

                    self.formulation += (
                        ufl.inner(lmbda * ufl.dot(vel, self.mesh.n) * u, v) * self.ds
                    )

        for adv_term in self.advection_terms:
            for species in adv_term.species:
                conc = species.solution
                v = species.test_function
                vel = adv_term.velocity.fenics_object

                lmbda = ufl.conditional(ufl.gt(ufl.dot(vel, self.mesh.n), 0), 1, 0)

                # Advection, bulk
                self.formulation += -ufl.inner(vel * conc, ufl.grad(v)) * self.dx(
                    adv_term.subdomain.id
                )

                # Advection, interior upwind flux
                self.formulation += (
                    ufl.inner(2 * ufl.avg(lmbda * vel * conc), ufl.jump(v, self.mesh.n))
                    * self.dS
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
                                if vol in not_defined_in_volume:
                                    not_defined_in_volume.remove(vol)

                    # add c = 0 to formulation where needed
                    for vol in not_defined_in_volume:
                        self.formulation += (
                            spe.solution * spe.test_function * self.dx(vol.id)
                        )
