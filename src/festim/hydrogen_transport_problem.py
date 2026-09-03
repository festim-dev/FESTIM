import warnings
from collections.abc import Callable

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import numpy as np
import numpy.typing as npt
import scifem
import tqdm.auto
import ufl
from dolfinx import fem
from dolfinx.cpp.fem import compute_integration_domains
from dolfinx.fem.petsc import NonlinearProblem
from packaging.version import Version

from festim import (
    boundary_conditions,
    exports,
    k_B,
    problem,
)
from festim import (
    drift as _drift,
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
from festim.helpers import (
    restrict as _restrict,
)
from festim.mixed_dimensional_assembly import (
    custom_assemble_jacobian,
    custom_assemble_residual,
    make_index_sets,
    prune_empty_blocks,
)

from .mesh import CoordinateSystem, Mesh

__all__ = [
    "HydrogenTransportProblem",
    "HydrogenTransportProblemDiscontinuous",
]


def _warn_advection_terms_deprecated():
    warnings.warn(
        "advection_terms is deprecated, use drift_terms instead. An advection term is "
        "one kind of drift term, alongside festim.SoretTerm and "
        "festim.ElectromigrationTerm, and drift_terms holds all of them.",
        DeprecationWarning,
        stacklevel=3,
    )


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
        advection_terms: deprecated, use ``drift_terms``. Appended to it
        drift_terms: the drift terms of the model -- advection, Soret, electromigration

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
        advection_terms: deprecated, use ``drift_terms``. Appended to it
        drift_terms: the drift terms of the model -- advection, Soret, electromigration
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

    drift_terms: list[_drift.DriftTermBase]
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
        drift_terms=None,
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
        self.drift_terms = list(drift_terms or [])
        if advection_terms:
            # appended, not assigned through the deprecated setter, so that passing both
            # keeps every term the user gave
            _warn_advection_terms_deprecated()
            self.drift_terms += list(advection_terms)
        self.temperature_fenics = None

        self._unpacked_sources = []

        self._element_immobile = element_immobile

        self._temperature_as_function = None
        self._species_to_D_global = None
        self._species_to_D_global_expr = None
        self._surface_to_volume = None

    @property
    def advection_terms(self):
        """Deprecated. The advection terms among :attr:`drift_terms`.

        An advection term is one kind of drift term, alongside
        :class:`festim.SoretTerm` and :class:`festim.ElectromigrationTerm`, and they all
        live in ``drift_terms`` now. This reads back only the
        :class:`festim.AdvectionTerm` entries, so it is *not* an alias of the whole
        list.
        """
        _warn_advection_terms_deprecated()
        return [t for t in self.drift_terms if isinstance(t, AdvectionTerm)]

    @advection_terms.setter
    def advection_terms(self, value):
        _warn_advection_terms_deprecated()
        # replace the advection terms and keep every other kind, so that assigning here
        # cannot silently drop a Soret or electromigration term the user set separately
        self.drift_terms = [
            t for t in self.drift_terms if not isinstance(t, AdvectionTerm)
        ] + list(value or [])

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
        self.convert_drift_terms_to_fenics_objects()
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

    def _export_context(
        self, export
    ) -> tuple[list[fem.Function], list[str], dolfinx.mesh.Mesh]:
        """Resolve what a field export should write.

        Args:
            export: the export to resolve

        Returns:
            the functions to write, the name to store each one under, and the mesh
            they live on
        """
        if isinstance(export, exports.TemperatureExport):
            self._temperature_as_function = self._get_temperature_field_as_function()
            return (
                [self._temperature_as_function],
                [self._temperature_as_function.name],
                self._temperature_as_function.function_space.mesh,
            )
        elif isinstance(export, exports.SpeciesExport):
            functions = export.get_functions()
            names = [species.name for species in export.field]
            return functions, names, self.mesh.mesh
        elif isinstance(export, exports.CustomFieldExport):
            export.function = fem.Function(self.V_CG_1)
            export.set_dolfinx_expression(
                temperature=self.temperature_fenics,
                time=self.t,
            )
            return (
                [export.function],
                [export.filename.stem],
                export.function.function_space.mesh,
            )
        raise NotImplementedError(f"Export type {type(export)} not implemented")

    def initialise_exports(self):
        """Defines the export writers of the model, if field is given as a string, find
        species object in self.species."""

        # formats that hold several meshes in one file (vtkhdf) let exports share a
        # filename: the first one to claim it truncates, the rest append as new blocks
        initialised_files = set()
        for export in self.exports:
            # if name of species is given then replace with species object. Done first
            # so that the writers below get real Species to work from.
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

            if isinstance(export, exports.FieldExportBase):
                self._register_export_milestones(export)
                functions, names, mesh = self._export_context(export)
                # everything lives on the parent mesh here, so a single block per file
                export.define_writer(
                    functions,
                    names,
                    mesh,
                    overwrite=export.filename not in initialised_files,
                )
                initialised_files.add(export.filename)

            elif isinstance(export, exports.DerivedQuantity):
                # raise not implemented error if the derived quantity don't match the
                # type of mesh eg. SurfaceFlux is used with cylindrical mesh
                if self.mesh.coordinate_system != CoordinateSystem.CARTESIAN:
                    raise NotImplementedError(
                        f"Derived quantity exports are not implemented for "
                        f"{self.mesh.coordinate_system!s} meshes"
                    )

                # a volume subdomain is accepted as a surface only to let a codim-1
                # one (a manifold) through, and manifolds exist only in
                # HydrogenTransportProblemDiscontinuous. Here the id would be looked
                # up in the facet tags, silently reporting some other surface's value
                if isinstance(export, exports.SurfaceQuantity) and isinstance(
                    export.surface, _subdomain.VolumeSubdomain
                ):
                    raise TypeError(
                        f"volume subdomain {export.surface.id} was given as the "
                        f"surface of a {type(export).__name__}. Co-dim volume "
                        "subdomains are only supported by "
                        "HydrogenTransportProblemDiscontinuous"
                    )

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
            # a model without drift terms must not pay for the surface-to-volume
            # lookup, which needs meshtags this one does not otherwise require
            if self.drift_terms and isinstance(export, exports.SurfaceFlux):
                export.drift_velocity = self.drift_velocity_in(
                    field=export.field,
                    volume=self.volume_subdomain_of_surface(export.surface),
                )
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

    def drift_velocity_in(
        self,
        field: _species.Species,
        volume: _subdomain.VolumeSubdomain,
        temperature=None,
        mesh=None,
    ):
        """The summed drift velocity acting on ``field`` in ``volume``.

        Used in two places, both to do with the boundary term the divergence form
        leaves behind: :class:`festim.SurfaceFlux` has to report it, and
        :class:`festim.OutflowBC` cancels it.

        Args:
            field: the species whose flux is being computed
            volume: the volume subdomain the flux leaves
            temperature: the temperature on the mesh the integral is taken on, defaults
                to the problem's
            mesh: the mesh the integral is taken on, defaults to the problem's

        Returns:
            the summed drift velocity as a ufl expression, or ``None`` when no drift
            term acts on ``field`` in ``volume``
        """
        temperature = self.temperature_fenics if temperature is None else temperature
        mesh = self.mesh.mesh if mesh is None else mesh

        velocities = []
        for term in self.drift_terms:
            if term.subdomain is not volume:
                continue
            if field not in term.species:
                continue
            D = volume.material.get_diffusion_coefficient(mesh, temperature, field)
            velocity = term.drift_velocity(D=D, temperature=temperature)
            if not _drift.is_zero_velocity(velocity):
                velocities.append(velocity)

        if not velocities:
            return None
        return sum(velocities[1:], velocities[0])

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
                surface_subdomains=self.facet_surface_subdomains,
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

    def convert_drift_terms_to_fenics_objects(self):
        """For each drift term convert its user-given coefficients.

        Runs after ``define_temperature`` so that a coefficient given as a function of
        ``T``, or a drift velocity built from the temperature gradient, has one to read.
        """

        for drift_term in self.drift_terms:
            drift_term.convert_inputs(
                function_space=self.function_space,
                t=self.t,
                temperature=self.temperature_fenics,
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

        for drift_term in self.drift_terms:
            vol = drift_term.subdomain
            for species in drift_term.species:
                D = vol.material.get_diffusion_coefficient(
                    self.mesh.mesh, self.temperature_fenics, species
                )
                velocity = drift_term.drift_velocity(
                    D=D, temperature=self.temperature_fenics
                )
                # a term whose velocity is identically zero is still assembled: it costs
                # nothing (UFL folds the zero into the integrand it shares a measure
                # with) and dropping it would change the user's model behind their back
                _drift.warn_if_no_effect(drift_term, species, velocity)
                self.formulation += _drift.drift_form(
                    concentration=species.solution,
                    test_function=species.test_function,
                    velocity=velocity,
                    dx=self.dx(vol.id),
                    coordinate_system=self.mesh.coordinate_system,
                    mesh=self.mesh.mesh,
                )

        for bc in self.boundary_conditions:
            if isinstance(bc, boundary_conditions.OutflowBC):
                velocity = self.drift_velocity_in(
                    field=bc.species,
                    volume=self.volume_subdomain_of_surface(bc.subdomain),
                )
                if velocity is not None:
                    self.formulation += (
                        bc.species.solution
                        * ufl.dot(velocity, ufl.FacetNormal(self.mesh.mesh))
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

        for drift_term in self.drift_terms:
            drift_term.update_time_dependent_inputs(t=t)

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

            # handle field exports
            if isinstance(export, exports.FieldExportBase):
                if isinstance(export, exports.TemperatureExport):
                    if not self.temperature_time_dependent:
                        # nothing changed since the last write
                        continue
                    self._temperature_as_function.interpolate(
                        self._get_temperature_field_as_function()
                    )
                export.update()
                export.write(float(self.t))

            # TODO if export type derived quantity
            if isinstance(export, exports.SurfaceQuantity):
                if isinstance(
                    export,
                    exports.SurfaceFlux | exports.TotalSurface | exports.AverageSurface,
                ):
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
    # None unless the user sets the deprecated problem-level attribute; only then
    # does initialise() push it onto the interfaces (see method_interface)
    _method_interface: _subdomain.interface.InterfaceMethod | None = None
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
        advection_terms=None,
        drift_terms=None,
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
            advection_terms,
            drift_terms=drift_terms,
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
        if self._method_interface is None:
            return _subdomain.interface.InterfaceMethod.penalty
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
        # each interface and raise a deprecation warning. ``hasattr`` cannot be used to
        # detect that: method_interface is a property, so it is always present
        if self._method_interface is not None:
            warnings.warn(
                "The method_interface attribute of the Problem class is deprecated, "
                "please set the method_interface attribute of each interface instead",
                DeprecationWarning,
            )
            for interface in self.interfaces:
                interface.method = self._method_interface

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
                # a codim-2 surface bounds a manifold and is not in the facet tags; the
                # manifold it belongs to comes from the species of the bc using it
                surface_subdomains=self.facet_surface_subdomains,
                comm=self.mesh.mesh.comm,
            )

        # a manifold may sit on the boundary of the domain (one adjacent volume), on an
        # interface (two), or thread a polycrystal of one subdomain per grain (as many
        # as it touches), so it needs a multi-valued mapping of its own rather than the
        # one-to-one surface_to_volume
        facet_to_cell = self.mesh.mesh.topology.connectivity(
            self.mesh.mesh.topology.dim - 1, self.mesh.mesh.topology.dim
        )
        self.interior_facet_measure = None
        self._manifold_is_interior = {}
        self._manifold_export_measures = {}
        self._manifold_side_ids = {}
        self.manifold_to_volumes = _subdomain.map_manifold_to_volume_subdomains(
            ft=self.facet_meshtags,
            ct=self.volume_meshtags,
            facet_to_cell=facet_to_cell,
            volume_subdomains=self.volume_subdomains,
            manifold_subdomains=self.manifold_subdomains,
            comm=self.mesh.mesh.comm,
        )
        # eagerly, so that a manifold straddling the boundary of the mesh is rejected
        # when it is declared rather than only if something happens to couple to it
        for manifold in self.manifold_subdomains:
            self.manifold_is_interior(manifold)

        # create submeshes and transfer meshtags to subdomains
        for subdomain in self.volume_subdomains:
            if subdomain.codim(self.mesh.vdim) == 1:
                subdomain.create_subdomain(self.mesh.mesh, self.facet_meshtags)
            else:
                subdomain.create_subdomain(self.mesh.mesh, self.volume_meshtags)
            subdomain.transfer_meshtag(self.mesh.mesh, self.facet_meshtags)

        for interface in self.interfaces:
            # ``mt`` is read as ``mt.find(interface.id)`` to get the facets the
            # interface occupies, so it is the facet tags
            interface.mt = self.facet_meshtags
            interface.mesh = self.mesh.mesh
            interface.parent_mesh = self.mesh.mesh

        # every interior-facet integrand of the parent mesh is known by now, and they
        # all have to share one measure (see :meth:`build_interior_facet_measure`)
        self.build_interior_facet_measure()

        self.create_species_from_traps()
        self.link_enclosures()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self._dt = as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )

        self.create_submesh_time_constants()

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
        self.convert_drift_terms_to_fenics_objects()
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
        else:
            # a manifold subdomain integrates its gradient terms on its own submesh,
            # and a fem.Constant bound to the parent mesh cannot appear there
            for subdomain in self.manifold_subdomains:
                subdomain.sub_T = as_fenics_constant(
                    float(self.temperature_fenics), subdomain.submesh
                )

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
        on_manifold = bc.subdomain.codim(self.mesh.vdim) == 2
        if on_manifold:
            # the boundary of a manifold carries no meshtag: which manifold it bounds
            # follows from the species, exactly as a flux on an interior manifold picks
            # its side from bc.species
            volume_subdomain = self.manifold_of(bc.subdomain, bc.species)
            fdim = volume_subdomain.submesh.topology.dim - 1
        else:
            volume_subdomain = self.surface_to_volume[bc.subdomain]
            fdim = self.mesh.mesh.topology.dim - 1
        sub_V = bc.species.subdomain_to_function_space[volume_subdomain]
        collapsed_V, _ = sub_V.collapse()

        # in the discontinuous case, if the temperature is given as a function
        # then we can't use the temperature on the parent mesh
        # see issue #1007
        if on_manifold:
            # the value is interpolated onto the manifold's submesh, so the coefficients
            # it is built from have to live there too
            temp = volume_subdomain.sub_T
            time = volume_subdomain.sub_t
        elif isinstance(self.temperature_fenics, fem.Function):
            temp = volume_subdomain.sub_T
            time = self.t
        else:
            temp = self.temperature_fenics
            time = self.t

        bc.create_value(
            temperature=temp,
            function_space=collapsed_V,
            t=time,
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

        if on_manifold:
            # NOTE: this means that the only way to apply a DirichletBC on a manifold is
            # to be able to describe it geometrically. This is limiting and it would
            # be desirable to have a "meshtag route" for this.

            # located directly on the manifold's submesh -- no tag to look up, and no
            # codim-2 entity of the parent mesh is ever needed
            entities = bc.subdomain.locate_boundary_facet_indices(
                volume_subdomain.submesh
            )
            # a locator that matches nothing, or matches only points interior to the
            # manifold, would otherwise leave the bc silently doing nothing
            if self.mesh.mesh.comm.allreduce(len(entities), op=MPI.SUM) == 0:
                raise ValueError(
                    f"the locator of surface subdomain {bc.subdomain.id} matched no "
                    f"boundary entity of codim-1 volume subdomain "
                    f"{volume_subdomain.id}. It must select a point on the boundary of "
                    "that subdomain, not one inside it."
                )
        else:
            entities = volume_subdomain.ft.find(bc.subdomain.id)

        bc_dofs = dolfinx.fem.locate_dofs_topological(
            function_space_dofs,
            fdim,
            entities,
        )
        form = dolfinx.fem.dirichletbc(bc.value_fenics, bc_dofs, sub_V)
        return form

    def manifold_of(
        self, surface: _subdomain.SurfaceSubdomain, species: _species.Species
    ) -> _subdomain.VolumeSubdomain:
        """The manifold subdomain that a codim-2 ``surface`` bounds.

        A codim-2 surface is not in any meshtag, so it cannot be mapped to its volume
        topologically the way an ordinary surface is. It is instead resolved from the
        species of the term using it -- the same rule that gives a flux on an interior
        manifold its side. That also means one surface object may be reused on several
        manifolds, one species each.

        Raises:
            ValueError: if the species does not live on exactly one manifold subdomain
        """
        candidates = [s for s in self.manifold_subdomains if s in species.subdomains]
        if len(candidates) != 1:
            raise ValueError(
                f"cannot tell which manifold surface subdomain {surface.id} bounds: "
                f"species {species.name} lives on volume subdomains "
                f"{[s.id for s in species.subdomains]}, of which "
                f"{[s.id for s in candidates]} are manifolds. A codim-2 surface "
                "subdomain is resolved through the species of the boundary condition "
                "using it, so that species must live on exactly one manifold."
            )
        return candidates[0]

    def create_initial_conditions(self):
        """For each intial condition, create the value_fenics and assign it to the
        previous solution of the condition's species."""

        for condition in self.initial_conditions:
            # the index within the subdomain's mixed element, which is not the index of
            # the species in the global list unless every species lives on every
            # subdomain
            idx = condition.species.subdomain_to_index[condition.volume]

            # if the value given is a function, then directly interpolate it on the
            # previous solution of the species
            if isinstance(condition.value, fem.Function):
                nmm_interpolate(condition.volume.u_n.sub(idx), condition.value)

            else:
                V = condition.species.subdomain_to_function_space[condition.volume]

                if isinstance(self.temperature_fenics, fem.Function):
                    temperature = condition.volume.sub_T
                else:
                    temperature = self.temperature_fenics

                condition.create_expr_fenics(
                    mesh=condition.volume.submesh,
                    temperature=temperature,
                    function_space=V,
                )

                # assign to previous solution of species; the expression already
                # lives on the subdomain's submesh, so no cell restriction is needed
                condition.volume.u_n.sub(idx).interpolate(condition.expr_fenics)

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
            species.subdomain_to_index[subdomain] = i
            species.subdomain_to_post_processing_solution[subdomain] = u.sub(
                i
            ).collapse()
            species.subdomain_to_collapsed_function_space[subdomain] = V.sub(
                i
            ).collapse()
            name = f"{species.name}_{subdomain.id}"
            species.subdomain_to_post_processing_solution[subdomain].name = name

    def source_integration_mesh(self, source) -> dolfinx.mesh.Mesh:
        """The mesh the integral carrying ``source`` is assembled over.

        A source involving only fields of a manifold subdomain is integrated over that
        manifold's submesh (:meth:`subdomain_measure`); one coupling the manifold to
        the bulk is a facet integral of the parent mesh (:meth:`facet_measure`).
        """
        if self.is_manifold_self_source(source):
            return source.volume.submesh
        if source.volume in self.manifold_to_volumes:
            return self.mesh.mesh
        return source.species.subdomain_to_function_space[source.volume].mesh

    def convert_source_input_values_to_fenics_objects(self):
        """For each source create the value_fenics."""
        for source in self._unpacked_sources:
            if isinstance(source, _source.ParticleSource):
                # a self source on a manifold is integrated over its submesh, so its
                # time and temperature must live there; a coupling source is integrated
                # on the parent mesh and keeps the parent-mesh ones
                if self.is_manifold_self_source(source):
                    t = self.subdomain_time(source.volume)
                    temperature = self.subdomain_temperature(source.volume)
                else:
                    t = self.t
                    temperature = self.temperature_fenics

                source.value.convert_input_value(
                    function_space=source.species.subdomain_to_function_space[
                        source.volume
                    ],
                    mesh=self.source_integration_mesh(source),
                    t=t,
                    temperature=temperature,
                    up_to_ufl_expr=True,
                    subdomain=source.volume,
                    foreign_subdomain=self.source_coupling_side(source),
                )

    def convert_drift_terms_to_fenics_objects(self):
        """As the base class, but on the function space of the term's own subdomain.

        Every coefficient of a submesh integral has to be built on that submesh -- FFCx
        cannot tabulate a parent-mesh coefficient on submesh cells -- so the temperature
        is the subdomain's own, as it is for reaction rates.
        """

        for drift_term in self.drift_terms:
            for spe in drift_term.species:
                V = spe.subdomain_to_function_space[drift_term.subdomain]
                drift_term.convert_inputs(
                    function_space=V,
                    t=self.t,
                    temperature=self.subdomain_temperature(drift_term.subdomain),
                )

    def define_boundary_conditions(self):
        for bc in self._unpacked_bcs:
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                bc._volume_subdomain = self.flux_bc_target(bc)[0]

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

    def subdomain_measure(self, subdomain: _subdomain.VolumeSubdomain):
        """The measure the *self* terms of ``subdomain`` are integrated over.

        Self terms are the ones that only involve fields living on ``subdomain``: the
        time derivative, diffusion, advection, reactions and sources that do not reach
        across meshes. For a manifold (codim-1) subdomain they are integrated over its
        own submesh, which is what makes ``ufl.grad`` mean the tangential gradient (see
        the docs and issue #1208) and what makes the formulation identical whether the
        manifold sits on the boundary of the domain or between two volume subdomains.

        Terms coupling a manifold to the bulk cannot use this measure -- a bulk field
        cannot be resolved inside a codim-1 integral -- and use
        :meth:`facet_measure` instead.
        """
        if subdomain.codim(self.mesh.vdim) == 1:
            return ufl.Measure("dx", domain=subdomain.submesh)
        return self.dx(subdomain.id)

    def manifold_is_interior(self, manifold: _subdomain.VolumeSubdomain) -> bool:
        """Whether ``manifold`` sits on interior facets of the mesh.

        This is a question about the *mesh*, not about how many volume subdomains the
        manifold happens to separate: a grain boundary network inside a single-phase
        polycrystal is interior even though the same subdomain lies on both sides of it.
        Deciding from the subdomain count instead would pick ``ds`` for such a manifold,
        and ``ds`` integrates to exactly zero over interior facets -- a coupling that
        silently does nothing.

        Raises:
            ValueError: if the manifold mixes interior and exterior facets, which would
                need two measures at once
        """
        if manifold not in self._manifold_is_interior:
            mesh = self.mesh.mesh
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)
            facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
            facets = self.facet_meshtags.find(manifold.id)
            # vectorised: the number of cells a facet connects to is the width of
            # its slice in the adjacency list, and there can be tens of thousands
            # of facets in a manifold
            offsets = facet_to_cell.offsets
            n_cells = offsets[facets + 1] - offsets[facets]
            n_interior = int(np.count_nonzero(n_cells == 2))

            comm = mesh.comm
            total = comm.allreduce(len(facets), op=MPI.SUM)
            interior = comm.allreduce(n_interior, op=MPI.SUM)
            if 0 < interior < total:
                raise ValueError(
                    f"codim-1 volume subdomain {manifold.id} has {interior} interior "
                    f"and {total - interior} exterior facets. A manifold must lie "
                    "wholly inside the mesh or wholly on its boundary; split it into "
                    "two subdomains."
                )
            self._manifold_is_interior[manifold] = interior > 0
        return self._manifold_is_interior[manifold]

    def facet_measure(self, manifold: _subdomain.VolumeSubdomain):
        """The parent-mesh measure the facets of ``manifold`` are integrated in: the
        terms coupling it to the bulk, and the derived quantities exported on it.

        A manifold on the boundary of the mesh is integrated with ``ds``, one inside
        the mesh with the shared ``dS`` (see :meth:`build_interior_facet_measure`),
        whose entities are ordered so that ``"+"`` is the first of the two volume
        subdomains it separates (see :meth:`restriction_of`).

        The measure is **not** restricted to the manifold: index it with
        ``manifold.id``, as the derived quantities do with the parent ``ds``. Left
        unindexed in a form it integrates over everything the measure carries data
        for -- every tagged facet of the mesh for ``ds``, every interior manifold and
        interface for the shared ``dS``.
        """
        if not self.manifold_is_interior(manifold):
            return self.ds
        return self.interior_facet_measure

    def build_interior_facet_measure(self):
        """Builds the single ``dS`` measure that every interior-facet integral of the
        parent mesh shares, and stores it in ``interior_facet_measure``.

        The coupling terms of an interior manifold and the continuity terms of an
        :class:`festim.Interface` are both ``dS`` integrals of the parent mesh, and
        both need integration data of their own so that ``"+"`` and ``"-"`` land on the
        sides they are meant to. They cannot each carry their own measure: UFL collects
        one ``subdomain_data`` entry per integral of a form, and ``dolfinx.fem.form``
        asserts that they are all the *same* object before using the first of them for
        every id. A measure per manifold therefore breaks as soon as one volume
        subdomain touches two interior manifolds, or an interface and an interior
        manifold -- both integrals end up in that subdomain's ``F``. So the data of
        every interior-facet integral goes into one list, and the measure built from it
        is handed to all of them, each indexing it by its own id.
        """
        self._allocate_manifold_side_ids()
        integral_data = [
            entry
            for manifold in self.manifold_subdomains
            if self.manifold_is_interior(manifold)
            for entry in self._manifold_integration_data(manifold)
        ]
        integral_data += [
            interface.compute_mapped_interior_facet_data(self.volume_meshtags)
            for interface in self.interfaces
        ]
        self.interior_facet_measure = ufl.Measure(
            "dS", domain=self.mesh.mesh, subdomain_data=integral_data
        )

    def _allocate_manifold_side_ids(self):
        """Gives each side of a manifold adjacent to more than two volume subdomains an
        integration id of its own, in ``_manifold_side_ids``.

        Two sides fit in one integral: the facets are ordered once and the two coupling
        terms are told apart by ``"+"`` and ``"-"``. Beyond two there is no such
        ordering, because the facets of the manifold no longer separate the same pair of
        subdomains all the way along -- a grain-boundary network runs between a
        different pair of grains on either side of every triple junction. So each
        adjacent volume gets one integral over the facets it touches, ordered so that it
        is on ``"+"`` (see
        :func:`festim.subdomain.compute_one_sided_interior_facet_data`), and a facet
        shared by two grains is integrated once per grain.

        The ids are allocated above every id the user has declared, so that they cannot
        collide with another manifold, an interface or a surface in the shared ``dS``
        measure.
        """
        self._manifold_side_ids = {}
        declared = {v.id for v in self.volume_subdomains}
        declared |= {s.id for s in self.surface_subdomains}
        declared |= {i.id for i in self.interfaces}
        next_id = max(declared, default=0) + 1
        for manifold in self.manifold_subdomains:
            volumes = self.manifold_to_volumes[manifold]
            if len(volumes) <= 2:
                continue
            self._manifold_side_ids[manifold] = {}
            for volume in volumes:
                self._manifold_side_ids[manifold][volume] = next_id
                next_id += 1

    def _manifold_integration_data(self, manifold: _subdomain.VolumeSubdomain):
        """The ``(id, entities)`` pairs putting the coupling terms of an interior
        ``manifold`` on the facets it occupies.

        One pair, tagged with the manifold's own id, while it is adjacent to at most two
        volume subdomains. When it separates two of them the entities are ordered so
        that ``"+"`` is the first (see :meth:`restriction_of`); when the same subdomain
        lies on both sides there is nothing to order, because the bulk field is
        single-valued across the facet.

        Beyond two adjacent subdomains there is one pair *per side*, tagged with the ids
        allocated by :meth:`_allocate_manifold_side_ids`.
        """
        volumes = self.manifold_to_volumes[manifold]
        if len(volumes) > 2:
            facets = self.facet_meshtags.find(manifold.id)
            return [
                (
                    self._manifold_side_ids[manifold][volume],
                    _subdomain.compute_one_sided_interior_facet_data(
                        self.volume_meshtags, facets, volume
                    ),
                )
                for volume in volumes
            ]
        if len(volumes) == 2:
            return [
                _subdomain.compute_ordered_interior_facet_data(
                    self.volume_meshtags,
                    self.facet_meshtags,
                    manifold.id,
                    volumes[0],
                    volumes[1],
                )
            ]
        # one subdomain on both sides: either restriction reads the same value, so the
        # entities are taken in whatever order DOLFINx gives them
        return [
            (
                manifold.id,
                compute_integration_domains(
                    dolfinx.fem.IntegralType.interior_facet,
                    self.mesh.mesh.topology._cpp_object,
                    self.facet_meshtags.find(manifold.id),
                ),
            )
        ]

    def coupling_measure_id(
        self, manifold: _subdomain.VolumeSubdomain, volume: _subdomain.VolumeSubdomain
    ) -> int:
        """The id the terms coupling ``manifold`` to ``volume`` index the measure of
        :meth:`facet_measure` with.

        The manifold's own id while it is adjacent to at most two volume subdomains --
        one integral carries both sides, told apart by their restriction. Beyond that
        each side has an integral of its own (see :meth:`_allocate_manifold_side_ids`).
        """
        side_ids = self._manifold_side_ids.get(manifold)
        return manifold.id if side_ids is None else side_ids[volume]

    def coupling_measure(
        self, manifold: _subdomain.VolumeSubdomain, volume: _subdomain.VolumeSubdomain
    ):
        """The measure, indexed, that the terms coupling ``manifold`` to ``volume`` are
        integrated over. Pair it with :meth:`restriction_of`."""
        return self.facet_measure(manifold)(self.coupling_measure_id(manifold, volume))

    def restriction_of(
        self, manifold: _subdomain.VolumeSubdomain, volume: _subdomain.VolumeSubdomain
    ) -> str | None:
        """Which side of an interior ``manifold`` the subdomain ``volume`` is on.

        ``None`` when the manifold is on the boundary of the mesh, where the coupling is
        an exterior-facet integral and nothing needs restricting. ``"+"`` when the same
        subdomain lies on both sides: the bulk field is continuous across the facet, so
        both restrictions read the same value and the exchange is applied once. ``"+"``
        again when the manifold is adjacent to more than two subdomains, where each side
        has an integral of its own on which it is the ``"+"`` one -- so this must always
        be read together with :meth:`coupling_measure_id`.
        """
        if not self.manifold_is_interior(manifold):
            return None
        volumes = self.manifold_to_volumes[manifold]
        if len(volumes) != 2:
            return "+"
        return "+" if volume is volumes[0] else "-"

    def restrict(self, expression, restriction: str | None):
        """Apply ``restriction`` to a whole expression, or return it unchanged when the
        coupling is on exterior facets."""
        return _restrict(expression, restriction)

    def coupling_side(
        self, manifold: _subdomain.VolumeSubdomain, species: _species.Species
    ) -> _subdomain.VolumeSubdomain:
        """The volume subdomain on the side of ``manifold`` that ``species`` lives on.

        This is how a coupling term declares which side of an interface it applies to:
        a flux boundary condition or a source names a bulk species, and that species
        identifies the side. Nothing else has to be specified by the user.

        Raises:
            ValueError: if the species does not live on exactly one of the manifold's
                adjacent volume subdomains
        """
        candidates = [
            v for v in self.manifold_to_volumes[manifold] if v in species.subdomains
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"cannot tell which side of codim-1 subdomain {manifold.id} species "
                f"{species.name} is on: it lives on volume subdomains "
                f"{[s.id for s in species.subdomains]} while {manifold.id} is adjacent "
                f"to {[v.id for v in self.manifold_to_volumes[manifold]]}. Declare one "
                "term per side, each naming the species of that side."
            )
        return candidates[0]

    def outflow_form(self, bc, subdomain: _subdomain.VolumeSubdomain):
        """The contribution of an :class:`festim.OutflowBC` to ``subdomain``'s form.

        Two positions. On an ordinary surface the integral is the parent ``ds`` of the
        volume the surface bounds. On the **boundary of a manifold** -- the outlet of a
        1D fluid, the case this exists for -- everything lives on that manifold's
        submesh: the measure, the normal, and the velocity, which was already built
        there when the drift term's inputs were converted.

        Args:
            bc: the outflow boundary condition
            subdomain: the volume subdomain whose form is being built

        Returns:
            the ufl form, or ``None`` if this bc does not belong to ``subdomain`` or no
            drift acts on its species there
        """
        if bc.subdomain.codim(self.mesh.vdim) == 2:
            manifold = self.manifold_of(bc.subdomain, bc.species)
            if manifold is not subdomain:
                return None
            mesh = manifold.submesh
            measure = self._manifold_boundary_measure(bc.subdomain, manifold)
            temperature = self.subdomain_temperature(manifold)
        else:
            if self.surface_to_volume[bc.subdomain] is not subdomain:
                return None
            mesh = self.mesh.mesh
            measure = self.ds
            temperature = self.temperature_fenics

        velocity = self.drift_velocity_in(
            field=bc.species, volume=subdomain, temperature=temperature, mesh=mesh
        )
        if velocity is None:
            return None

        return (
            bc.species.subdomain_to_solution[subdomain]
            * ufl.dot(velocity, ufl.FacetNormal(mesh))
            * bc.species.subdomain_to_test_function[subdomain]
            * measure(bc.subdomain.id)
        )

    def flux_bc_target(self, bc):
        """Where a ``ParticleFluxBC`` contributes: ``(volume subdomain, measure,
        restriction)``.

        On an ordinary surface this is the single volume the surface belongs to. On a
        manifold subdomain it is the side ``bc.species`` lives on, which is how a user
        declares one flux per side of an interface without naming the side explicitly.
        """
        if bc.subdomain in self.manifold_to_volumes:
            target = self.coupling_side(bc.subdomain, bc.species)
            return (
                target,
                self.coupling_measure(bc.subdomain, target),
                self.restriction_of(bc.subdomain, target),
            )
        return self.surface_to_volume[bc.subdomain], self.ds(bc.subdomain.id), None

    def foreign_species(
        self, source, manifold: _subdomain.VolumeSubdomain
    ) -> list[_species.Species]:
        """The species a source on ``manifold`` reads that do not live on it.

        A non-empty result means the source is one half of a codimensional coupling
        rather than a source of the manifold's own equation, and must therefore be
        integrated on the parent mesh.

        Raises:
            ValueError: if the source reads species from more than one side of the
                manifold, which cannot be expressed as a single restricted integral
        """
        dependencies = getattr(source.value, "species_dependent_value", None) or {}
        foreign = [
            species
            for species in dependencies.values()
            if manifold not in species.subdomains
        ]
        sides = {self.coupling_side(manifold, species) for species in foreign}
        if len(sides) > 1:
            raise ValueError(
                f"the source on codim-1 subdomain {manifold.id} depends on species "
                "from "
                f"both of its sides ({sorted(v.id for v in sides)}), which cannot be "
                "integrated as one term. Declare one source per side."
            )
        return foreign

    def create_submesh_time_constants(self):
        """Mirrors ``t`` and ``dt`` onto the submesh of every manifold subdomain.

        The self terms of a manifold are integrated over its own submesh (see
        :meth:`subdomain_measure`) and every coefficient of such an integral must live
        on that submesh: a ``fem.Constant`` bound to the parent mesh makes the integral
        mixed-dimensional and fails inside FFCx with an undiagnosable
        ``UnboundLocalError``. The mirrors are kept in sync by
        :meth:`update_submesh_time_constants`.
        """
        for subdomain in self.manifold_subdomains:
            subdomain.sub_t = as_fenics_constant(float(self.t), subdomain.submesh)
            if self.settings.transient:
                subdomain.sub_dt = as_fenics_constant(float(self.dt), subdomain.submesh)

    def update_submesh_time_constants(self):
        """Copies the current ``t`` and ``dt`` into the submesh mirrors created by
        :meth:`create_submesh_time_constants`, so that an adaptive stepsize and
        explicitly time-dependent values on a manifold see the same values as the
        rest of the problem."""
        for subdomain in self.manifold_subdomains:
            subdomain.sub_t.value = float(self.t)
            if self.settings.transient:
                subdomain.sub_dt.value = float(self.dt)

    def subdomain_time(self, subdomain: _subdomain.VolumeSubdomain):
        """The time constant to use in the self terms of ``subdomain``."""
        if subdomain.codim(self.mesh.vdim) == 1:
            return subdomain.sub_t
        return self.t

    def subdomain_dt(self, subdomain: _subdomain.VolumeSubdomain):
        """The timestep constant to use in the self terms of ``subdomain``."""
        if subdomain.codim(self.mesh.vdim) == 1:
            return subdomain.sub_dt
        return self.dt

    def subdomain_temperature(self, subdomain: _subdomain.VolumeSubdomain):
        """The temperature to use in the self terms of ``subdomain``."""
        if subdomain.codim(self.mesh.vdim) == 1:
            return subdomain.sub_T
        return self.temperature_fenics

    def convert_reaction_rates_to_fenics_objects(self):
        """As the base class, but with the temperature of the reaction's own mesh.

        A reaction on a manifold becomes a source integrated over that manifold's
        submesh, so its rate cannot be built from the parent-mesh temperature: FFCx
        cannot tabulate a parent-mesh coefficient on submesh cells.
        """
        for reaction in self.reactions:
            for rate in reaction.rate_coefficients:
                if rate.input_value is not None:
                    rate.convert_input_value(
                        function_space=getattr(self, "function_space", None),
                        t=self.t,
                        temperature=self.subdomain_temperature(reaction.volume),
                        subdomain=reaction.volume,
                        up_to_ufl_expr=True,
                    )

    def create_implicit_species_value_fenics(self):
        """For each implicit species, create the value_fenics.

        The density of an implicit species consumed by a reaction on a manifold
        subdomain appears in an integral over that manifold's submesh, so like every
        other coefficient of such an integral it has to be built there rather than on
        the parent mesh (see :meth:`create_submesh_time_constants`).
        """
        species_to_mesh = {}
        for reaction in self.reactions:
            volume = reaction.volume
            if volume is not None and volume.codim(self.mesh.vdim) == 1:
                mesh, t = volume.submesh, self.subdomain_time(volume)
            else:
                mesh, t = self.mesh.mesh, self.t

            for reactant in reaction.reactant:
                if not isinstance(reactant, _species.ImplicitSpecies):
                    continue

                # an implicit species shared by reactions on different meshes would be
                # built twice and keep only the last one, silently leaving a foreign
                # terminal/"entity" in one of the two integrals
                previous = species_to_mesh.setdefault(id(reactant), mesh)
                if previous is not mesh:
                    raise NotImplementedError(
                        f"implicit species {reactant.name} is used by reactions on a "
                        "codim-1 subdomain and on another subdomain, which are "
                        "integrated over different meshes. Declare a separate "
                        "implicit species for each subdomain."
                    )

                reactant.create_value_fenics(mesh=mesh, t=t)

                # a density given as a ready-made fenics object is passed through
                # untouched, so it is the one case building on the submesh cannot fix;
                # catch it here rather than let FFCx fail undiagnosably
                domain = ufl.domain.extract_unique_domain(reactant.value_fenics)
                if domain is not None and domain is not mesh.ufl_domain():
                    raise NotImplementedError(
                        f"the density of implicit species {reactant.name} is defined"
                        " on another mesh than the codim-1 subdomain "
                        f"{reaction.volume.id} its reaction is integrated over. Give "
                        "it as a float or as a callable of x and t instead of as a "
                        "ready-made fenics object."
                    )

    def is_manifold_self_source(self, source) -> bool:
        """Whether ``source`` belongs to a manifold's own equation, as opposed to being
        one half of a codimensional coupling or an ordinary volumetric source."""
        if source.volume not in self.manifold_to_volumes:
            return False
        return not self.foreign_species(source, source.volume)

    def source_coupling_side(self, source) -> _subdomain.VolumeSubdomain | None:
        """The bulk subdomain a coupling source on a manifold reads, or ``None`` if
        ``source`` is not one half of a codimensional coupling.

        Which side of the manifold the term belongs to is decided by the bulk species
        the source names (see :meth:`coupling_side`), and the expression has to read
        that species' solution *there*. It cannot work that out for itself: a bulk
        species defined on several subdomains -- because it is continuous across an
        interface, say -- has a solution on each of them.
        """
        if source.volume not in self.manifold_to_volumes:
            return None
        foreign = self.foreign_species(source, source.volume)
        if not foreign:
            return None
        # foreign_species has already checked that they all resolve to the same side
        return self.coupling_side(source.volume, foreign[0])

    def diffusion_coefficient(self, subdomain: _subdomain.VolumeSubdomain, species):
        """The diffusion coefficient of ``species`` for the gradient terms of
        ``subdomain``, defined on the mesh those terms are integrated over."""
        if subdomain.codim(self.mesh.vdim) == 1:
            # coefficients in a submesh integral must live on that submesh
            return subdomain.material.get_diffusion_coefficient(
                subdomain.submesh, subdomain.sub_T, species
            )
        return subdomain.material.get_diffusion_coefficient(
            self.mesh.mesh, self.temperature_fenics, species
        )

    def create_subdomain_formulation(self, subdomain: _subdomain.VolumeSubdomain):
        """Creates the variational formulation for each subdomain and stores it in
        ``subdomain.F`` and, for a manifold subdomain, ``subdomain.F_submesh``.

        Terms are split by whether they reach across meshes. *Self* terms -- the time
        derivative, diffusion, advection, reactions and sources involving only fields of
        ``subdomain`` -- are integrated over :meth:`subdomain_measure`, which for a
        manifold subdomain is its own submesh. *Coupling* terms, which mix a manifold
        field with a bulk field, cannot live there (a bulk function cannot be resolved
        inside a codim-1 integral) and use :meth:`facet_measure` on the parent mesh.

        For a regular volume subdomain both measures are on the parent mesh and the
        whole formulation ends up in ``subdomain.F``. For a manifold subdomain they are
        measures on two different meshes, which DOLFINx cannot compile into a single
        form, so the self terms are kept apart in ``subdomain.F_submesh`` and assembled
        as a separate group.

        Args:
            subdomain (F.VolumeSubdomain): a subdomain of the geometry
        """
        is_manifold = subdomain.codim(self.mesh.vdim) == 1
        if is_manifold and self.mesh.coordinate_system != CoordinateSystem.CARTESIAN:
            raise NotImplementedError(
                "codimensional subdomains are only supported in cartesian coordinates, "
                f"not {self.mesh.coordinate_system}"
            )
        dx = self.subdomain_measure(subdomain)
        # the self terms are integrated over subdomain's own mesh, so their coefficients
        # must live there too -- for a manifold that is its submesh, not the parent mesh
        dt = self.subdomain_dt(subdomain) if self.settings.transient else None

        self_form = 0
        form_coupling = 0
        # add diffusion and time derivative for each species
        for spe in self.species:
            if subdomain not in spe.subdomains:
                continue
            u = spe.subdomain_to_solution[subdomain]
            u_n = spe.subdomain_to_prev_solution[subdomain]
            v = spe.subdomain_to_test_function[subdomain]

            if self.settings.transient:
                self_form += ((u - u_n) / dt) * v * dx

            if spe.mobile:
                D = self.diffusion_coefficient(subdomain, spe)
                match self.mesh.coordinate_system:
                    case CoordinateSystem.CARTESIAN:
                        self_form += ufl.dot(D * ufl.grad(u), ufl.grad(v)) * dx
                    case CoordinateSystem.CYLINDRICAL:
                        r = ufl.SpatialCoordinate(self.mesh.mesh)[0]
                        self_form += r * ufl.dot(D * ufl.grad(u), ufl.grad(v / r)) * dx
                    case CoordinateSystem.SPHERICAL:
                        r = ufl.SpatialCoordinate(self.mesh.mesh)[0]
                        self_form += (
                            r**2 * ufl.dot(D * ufl.grad(u), ufl.grad(v / r**2)) * dx
                        )
                    case _:
                        raise ValueError(
                            f"Unsupported coordinate system {self.mesh.coordinate_system}"  # noqa: E501
                        )

        # add drift (advection, Soret, electromigration)
        for drift_term in self.drift_terms:
            if drift_term.subdomain != subdomain:
                continue

            for spe in drift_term.species:
                velocity = drift_term.drift_velocity(
                    D=self.diffusion_coefficient(subdomain, spe),
                    temperature=self.subdomain_temperature(subdomain),
                )
                # unlike the base class this one does skip a zero term, because
                # self_form may have no other integral: a manifold that carries drift
                # and nothing else would end up a form with no arguments, which
                # compiles but cannot be assembled
                if _drift.warn_if_no_effect(drift_term, spe, velocity):
                    continue
                # on a manifold both grad(c) and grad(w) are tangential, so the form
                # already picks out the tangential part of the velocity
                self_form += _drift.drift_form(
                    concentration=spe.subdomain_to_solution[subdomain],
                    test_function=spe.subdomain_to_test_function[subdomain],
                    velocity=velocity,
                    dx=dx,
                    coordinate_system=self.mesh.coordinate_system,
                    mesh=self.mesh.mesh,
                )

        # add fluxes. These are always parent-mesh integrals: a flux on a manifold
        # subdomain mixes the bulk and manifold fields, so it goes into form_coupling
        for bc in self._unpacked_bcs:
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                # check that the bc is applied on a surface belonging to this subdomain
                target, measure, restriction = self.flux_bc_target(bc)
                if subdomain == target:
                    v = bc.species.subdomain_to_test_function[subdomain]
                    form_coupling -= (
                        self.restrict(bc.value_fenics, restriction)
                        * self.restrict(v, restriction)
                        * measure
                    )
            if isinstance(bc, boundary_conditions.FixedConcentrationBC):
                # as for fluxes, only the subdomain owning the surface gets the term,
                # and its material is the one setting D there
                if bc.enforce_weakly and subdomain == self.surface_to_volume.get(
                    bc.subdomain
                ):
                    u = bc.species.subdomain_to_solution[subdomain]
                    v = bc.species.subdomain_to_test_function[subdomain]
                    D = subdomain.material.get_diffusion_coefficient(
                        self.mesh.mesh, self.temperature_fenics, bc.species
                    )
                    form_coupling += bc.weak_formulation(u, v, self.ds, D)

            # let drift carry the species out where the user says it flows out
            if isinstance(bc, boundary_conditions.OutflowBC):
                outflow = self.outflow_form(bc, subdomain)
                if outflow is not None:
                    self_form += outflow

        # add volumetric sources
        # reactions are expanded into particle sources (_unpacked_sources, see
        # create_sources_from_reactions), so they are handled by the source loop
        for source in self._unpacked_sources:
            if source.volume != subdomain:
                continue
            v = source.species.subdomain_to_test_function[subdomain]
            bulk = self.source_coupling_side(source) if is_manifold else None
            if bulk is not None:
                # a source on a manifold that reads a bulk concentration: the exchange
                # half that feeds the manifold. It must be integrated on the parent
                # mesh, and restricted to the side the bulk species lives on
                restriction = self.restriction_of(subdomain, bulk)
                form_coupling -= (
                    self.restrict(source.value.fenics_object, restriction)
                    * self.restrict(v, restriction)
                    * self.coupling_measure(subdomain, bulk)
                )
            else:
                self_form -= source.value.fenics_object * v * dx

        # store the form(s) in the subdomain object
        if is_manifold:
            subdomain.F = form_coupling
            # self_form is still the integer 0 if the manifold carries no equation of
            # its own, in which case there is nothing to assemble over the submesh
            subdomain.F_submesh = self_form if isinstance(self_form, ufl.Form) else None
        else:
            subdomain.F = self_form + form_coupling
            subdomain.F_submesh = None

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

    def interface_species(self, interface: _subdomain.Interface):
        """The mobile species whose continuity ``interface`` enforces.

        An interface condition relates the two solutions of one species across the
        facets, so it applies to a species that has a solution on both of its volume
        subdomains. In a model with codim-1 subdomains most species do not: a
        manifold's own species lives on a subdomain that is not a volume of any
        interface, and a bulk species may be confined to one side of a manifold. Those
        are simply not part of this interface's condition.

        A species present on exactly one of the two sides is the ambiguous case. It is
        either deliberately absent from the neighbouring material or a subdomain
        missing from its ``subdomains``, and only the user can tell which, so it is
        skipped with a warning rather than silently.
        """
        subdomain_0, subdomain_1 = interface.subdomains
        coupled = []
        for species in self.species:
            if not species.mobile:
                continue
            present = [s for s in (subdomain_0, subdomain_1) if s in species.subdomains]
            if len(present) == 2:
                coupled.append(species)
            elif present:
                missing = subdomain_1 if present[0] is subdomain_0 else subdomain_0
                warnings.warn(
                    f"species {species.name} lives on volume subdomain "
                    f"{present[0].id} but not on {missing.id}, the other side of "
                    f"interface {interface.id}, so the interface condition is not "
                    "applied to it. Add the missing subdomain to its `subdomains` if "
                    "it was meant to be continuous across that interface.",
                    stacklevel=2,
                )
        return coupled

    def create_formulation(self):
        """Takes all the formulations for each subdomain and adds the interface
        conditions.

        Finally compute the jacobian matrix and store it in the ``J`` attribute,
        adds the ``entity_maps`` to the forms and store them in the ``forms`` attribute
        """
        # the interfaces were wired and their integration data folded into the shared
        # dS measure in initialise(), so that an interface and an interior manifold
        # bounding the same volume subdomain still agree on their subdomain data
        dInterface = self.interior_facet_measure

        for interface in self.interfaces:
            F_0, F_1 = interface.get_formulation(
                dInterface,
                species=self.interface_species(interface),
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

        # a manifold subdomain contributes a second residual, integrated over its own
        # submesh; DOLFINx compiles one integration domain per form, so it is kept as a
        # separate group and summed into the residual and Jacobian at assembly time
        submesh_forms = [
            getattr(subdomain, "F_submesh", None)
            for subdomain in self.volume_subdomains
        ] + [None for _ in self.gas_species]
        groups = [all_forms]
        padded = set()
        if any(form is not None for form in submesh_forms):
            groups.append(submesh_forms)

            # the first group sets the block layout of the solver, so every block of it
            # must have a test space. A manifold with no coupling term contributes
            # nothing to it, and DOLFINx would then fail with "Could not deduce all
            # block test spaces"
            blocks = zip(all_forms, all_unknowns, strict=True)
            for i, (form, unknown) in enumerate(blocks):
                if not isinstance(form, ufl.Form):
                    v = ufl.TestFunction(unknown.function_space)
                    all_forms[i] = ufl.ZeroBaseForm((v,))
                    padded.add(i)

        # this is the symbolic differentiation of the Jacobian
        J_groups = []
        for group in groups:
            J = []
            for i, form in enumerate(group):
                # a padded row is zero by construction, and ufl.derivative of a
                # ZeroBaseForm cannot be expanded ("Rule not set for ZeroBaseForm")
                if form is None or (group is all_forms and i in padded):
                    J.append([None] * len(all_unknowns))
                    continue
                J.append([ufl.derivative(form, unknown) for unknown in all_unknowns])
            J_groups.append(J)
        if len(groups) > 1:
            # a block differentiated with respect to an unknown it does not depend on
            # would otherwise send DOLFINx looking for an entity map between two
            # sibling submeshes, which does not exist
            J_groups = [prune_empty_blocks(J) for J in J_groups]

            # the padded rows must still carry their function spaces: the blocked
            # matrix and the PETSc index sets are both built from this group alone, and
            # an all-None row leaves them with nothing to deduce the block from
            for i in padded:
                V = all_unknowns[i].function_space
                J_groups[0][i][i] = ufl.ZeroBaseForm(
                    (ufl.TestFunction(V), ufl.TrialFunction(V))
                )

        # compile jacobian (J) and residual (F)
        entity_maps = [sd.cell_map for sd in self.volume_subdomains]
        jit_options = {
            "cffi_extra_compile_args": ["-O3", "-march=native"],
            "cffi_libraries": ["m"],
        }

        self.form_groups = [
            dolfinx.fem.form(g, entity_maps=entity_maps, jit_options=jit_options)
            for g in groups
        ]
        self.J_groups = [
            dolfinx.fem.form(g, entity_maps=entity_maps, jit_options=jit_options)
            for g in J_groups
        ]
        self.forms = self.form_groups[0]
        self.J = self.J_groups[0]

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

        if len(self.form_groups) > 1:
            self._setup_mixed_dimensional_assembly()

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

    def _setup_mixed_dimensional_assembly(self):
        """Make the solver sum the residual and Jacobian over all form groups.

        ``NonlinearProblem`` assembles a single group; with a manifold subdomain the
        formulation spans two integration domains and therefore two groups, so the SNES
        callbacks are replaced by ones that accumulate both. The problem was built from
        the first (parent-mesh) group, which sets the block layout and the sparsity
        pattern.
        """
        snes = self.solver.solver
        unknowns = [subdomain.u for subdomain in self.volume_subdomains] + [
            gas_species.solution for gas_species in self.gas_species
        ]
        # the sparsity pattern comes from the first group alone, so allow the manifold
        # blocks to add nonzeros the bulk blocks did not need
        self.solver.A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        snes.setJacobian(
            custom_assemble_jacobian,
            self.solver.A,
            self.solver.P_mat,
            kargs={
                "u": unknowns,
                "jacobian_groups": self.J_groups,
                "preconditioner": None,
                "bcs": self.bc_forms,
                "index_sets": make_index_sets(self.J_groups[0]),
            },
        )

        residual_kargs = {
            "u": unknowns,
            "residual_groups": self.form_groups,
            "jacobian_groups": self.J_groups,
            "bcs": self.bc_forms,
        }
        if (blocks := self.solver.b.getAttr("_blocks")) is not None:
            residual_kargs["_blocks"] = blocks
        snes.setFunction(custom_assemble_residual, self.solver.b, kargs=residual_kargs)

    def create_flux_values_fenics(self):
        """For each particle flux create the ``value_fenics`` attribute."""
        for bc in self._unpacked_bcs:
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                bc.create_value_fenics(
                    mesh=self.mesh.mesh,
                    temperature=self.temperature_fenics,
                    t=self.t,
                )

    def export_volume_measure(self, volume: _subdomain.VolumeSubdomain):
        """The measure a volume quantity over ``volume`` is integrated with.

        A manifold's fields live on its own submesh, so its integrals are taken there --
        the parent ``dx`` carries no cell tagged with a manifold's id, and assembling a
        submesh field against a parent measure fails inside FFCx anyway. The measure is
        given a meshtag covering the whole submesh so that the export can index it by
        the subdomain id, exactly as it does with the parent ``dx``.
        """
        if volume.codim(self.mesh.vdim) != 1:
            return self.dx

        if volume not in self._manifold_export_measures:
            submesh = volume.submesh
            tdim = submesh.topology.dim
            # owned cells only: a ghost would be counted twice in parallel
            cells = np.arange(
                submesh.topology.index_map(tdim).size_local, dtype=np.int32
            )
            tags = dolfinx.mesh.meshtags(
                submesh, tdim, cells, np.full(len(cells), volume.id, dtype=np.int32)
            )
            self._manifold_export_measures[volume] = ufl.Measure(
                "dx", domain=submesh, subdomain_data=tags
            )
        return self._manifold_export_measures[volume]

    def _manifold_boundary_measure(
        self,
        surface: _subdomain.SurfaceSubdomain,
        manifold: _subdomain.VolumeSubdomain,
    ):
        """The ``ds`` measure on ``manifold``'s submesh selecting ``surface``.

        The boundary of a manifold is codim-1 *relative to the manifold*, so this is an
        ordinary exterior facet integral on its submesh rather than a codim-2 integral
        of the parent mesh -- the endpoints of a line in 2D are the facets of a 1D mesh.
        The same reasoning already locates the dofs of a Dirichlet BC there.

        Raises:
            ValueError: if the locator matches no boundary entity of the manifold
        """
        key = (surface, manifold)
        if key not in self._manifold_export_measures:
            submesh = manifold.submesh
            fdim = submesh.topology.dim - 1
            submesh.topology.create_connectivity(fdim, submesh.topology.dim)
            entities = surface.locate_boundary_facet_indices(submesh)
            # a locator matching nothing, or only points interior to the manifold, would
            # otherwise leave the export quietly reporting zero
            if self.mesh.mesh.comm.allreduce(len(entities), op=MPI.SUM) == 0:
                raise ValueError(
                    f"the locator of surface subdomain {surface.id} matched no "
                    f"boundary entity of codim-1 volume subdomain {manifold.id}. It "
                    "must select a point on the boundary of that subdomain, not one "
                    "inside it."
                )
            entities = np.sort(entities).astype(np.int32)
            tags = dolfinx.mesh.meshtags(
                submesh,
                fdim,
                entities,
                np.full(len(entities), surface.id, dtype=np.int32),
            )
            self._manifold_export_measures[key] = ufl.Measure(
                "ds", domain=submesh, subdomain_data=tags
            )
        return self._manifold_export_measures[key]

    def export_surface_context(self, export: exports.SurfaceQuantity):
        """Everything a surface quantity needs to know about where it is computed.

        Three cases, and the ordinary one is unchanged:

        - an ordinary ``SurfaceSubdomain``: the parent ``ds``, and the volume subdomain
          it bounds;
        - a **manifold** (codim-1 ``VolumeSubdomain``): the facets it occupies, read
          from the bulk side that ``export.field`` lives on. Interior manifolds use the
          ``dS`` coupling measure with that side's restriction -- the parent ``ds``
          integrates to exactly zero over interior facets, which would be a silent zero
          rather than an error;
        - a **codim-2 ``SurfaceSubdomain``**: the boundary of the manifold that
          ``export.field`` lives on, integrated on that manifold's submesh.

        Returns:
            ``(volume, mesh, measure, subdomain_id, restriction, entity_maps)`` where
            ``volume`` is the subdomain whose solution and material the quantity reads,
            ``mesh`` the one the integral is taken on -- the parent mesh except on the
            boundary of a manifold -- and ``subdomain_id`` the id to index ``measure``
            with, which is the surface's own except on a side of a manifold adjacent to
            more than two volume subdomains (see :meth:`coupling_measure_id`).

        Raises:
            ValueError: if the volume subdomain given as a surface is not a manifold, or
                if a manifold's own species is asked for on that manifold's facets,
                which is not a flux across anything
        """
        surface = export.surface
        entity_maps = [sd.cell_map for sd in self.volume_subdomains]
        parent = self.mesh.mesh

        if isinstance(surface, _subdomain.VolumeSubdomain):
            if surface.codim(self.mesh.vdim) != 1:
                raise ValueError(
                    f"volume subdomain {surface.id} is not a manifold, so it cannot be "
                    "used as the surface of a derived quantity. Only a codim-1 volume "
                    "subdomain occupies facets; give a surface subdomain, or a volume "
                    "quantity over this subdomain."
                )
            if surface in export.field.subdomains:
                raise ValueError(
                    f"species {export.field.name} lives on codim-1 volume subdomain "
                    f"{surface.id} itself, so it has no flux across it. Export it with "
                    "a volume quantity over that subdomain instead, or name a bulk "
                    "species to get the exchange with one side of it."
                )
            volume = self.coupling_side(surface, export.field)
            return (
                volume,
                parent,
                self.facet_measure(surface),
                self.coupling_measure_id(surface, volume),
                self.restriction_of(surface, volume),
                entity_maps,
            )

        if surface.codim(self.mesh.vdim) == 2:
            manifold = self.manifold_of(surface, export.field)
            # everything lives on the submesh, so no entity map is involved
            return (
                manifold,
                manifold.submesh,
                self._manifold_boundary_measure(surface, manifold),
                surface.id,
                None,
                None,
            )

        return (
            self.surface_to_volume[surface],
            parent,
            self.ds,
            surface.id,
            None,
            entity_maps,
        )

    def _export_context(
        self, export
    ) -> tuple[list[fem.Function], list[str], dolfinx.mesh.Mesh]:
        """Resolve what a field export should write.

        Species and custom fields live on submeshes here, so each export writes on the
        mesh of its own subdomain rather than on the parent mesh.

        Args:
            export: the export to resolve

        Returns:
            the functions to write, the name to store each one under, and the mesh
            they live on
        """
        if isinstance(export, exports.SpeciesExport):
            functions = export.get_functions()
            # NOTE: names are deliberately un-suffixed (`H`, not `H_1`) even though the
            # collapsed functions are named per subdomain: checkpoints written before
            # this refactor used the bare species name and are read back by name.
            names = [species.name for species in export.field]
            return functions, names, functions[0].function_space.mesh
        elif isinstance(export, exports.TemperatureExport):
            assert isinstance(self.temperature_fenics, fem.Function), (
                "Temperature must be space-dependent to be exported as "
                "TemperatureExport"
            )
            return (
                [self.temperature_fenics],
                [self.temperature_fenics.name],
                self.temperature_fenics.function_space.mesh,
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
            return (
                [export.function],
                [export.filename.stem],
                export.function.function_space.mesh,
            )
        raise NotImplementedError(f"Export type {type(export)} not implemented")

    def initialise_exports(self):
        # formats that hold several meshes in one file (vtkhdf) let exports share a
        # filename: the first one to claim it truncates, the rest append as new blocks
        initialised_files = set()
        for export in self.exports:
            if isinstance(export, exports.FieldExportBase):
                self._register_export_milestones(export)
                functions, names, mesh = self._export_context(export)
                # exports on different submeshes share a file as separate blocks
                subdomain = getattr(export, "subdomain", None)
                export.define_writer(
                    functions,
                    names,
                    mesh,
                    block_name=(
                        "mesh" if subdomain is None else f"subdomain_{subdomain.id}"
                    ),
                    overwrite=export.filename not in initialised_files,
                )
                initialised_files.add(export.filename)

        # compute diffusivity function for surface fluxes
        # for the discontinuous case, we don't use D_global as in
        # HydrogenTransportProblem
        for export in self.exports:
            if isinstance(export, exports.SurfaceQuantity):
                volume, mesh, *_ = self.export_surface_context(export)
                # on the boundary of a manifold the integral is taken on that manifold's
                # submesh, and like every other coefficient of a submesh integral D has
                # to be built there rather than on the parent mesh
                temperature = (
                    self.subdomain_temperature(volume)
                    if mesh is not self.mesh.mesh
                    else self.temperature_fenics
                )
                D = volume.material.get_diffusion_coefficient(
                    mesh, temperature, export.field
                )
                # NOTE: maybe we need to make sure there are no functionspace clashes?

                export.D = D

                if self.drift_terms and isinstance(export, exports.SurfaceFlux):
                    export.drift_velocity = self.drift_velocity_in(
                        field=export.field,
                        volume=volume,
                        temperature=temperature,
                        mesh=mesh,
                    )

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
            # handle field exports
            if isinstance(export, exports.FieldExportBase):
                export.update()
                export.write(float(self.t))
            # handle derived quantities
            if isinstance(export, exports.SurfaceQuantity):
                if isinstance(
                    export,
                    exports.SurfaceFlux | exports.TotalSurface | exports.AverageSurface,
                ):
                    export_vol, _, measure, subdomain_id, restriction, entity_maps = (
                        self.export_surface_context(export)
                    )
                    submesh_function = (
                        export.field.subdomain_to_post_processing_solution[export_vol]
                    )

                    export.compute(
                        u=submesh_function,
                        ds=measure,
                        entity_maps=entity_maps,
                        restriction=restriction,
                        subdomain_id=subdomain_id,
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
                        dx=self.export_volume_measure(export.volume),
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

        self.update_submesh_time_constants()

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
            else:
                # a manifold mirrors a constant temperature onto its own submesh, and
                # that mirror has to follow the parent constant
                for subdomain in self.manifold_subdomains:
                    # NOTE: current limitation for manifolds, temperature on the manifold
                    # has to be homogeneous (ie. fem.Constant)
                    subdomain.sub_T.value = float(self.temperature_fenics)

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
            if isinstance(export, exports.FieldExportBase):
                export.close()


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

        # add drift (advection, Soret, electromigration). The solved variable of a
        # mobile species is the chemical potential here, so the drift term is written
        # in terms of the concentration u * K_S, as the diffusion term above is
        for drift_term in self.drift_terms:
            vol = drift_term.subdomain
            for spe in drift_term.species:
                D = vol.material.get_diffusion_coefficient(
                    self.mesh.mesh, self.temperature_fenics, spe
                )
                conc = spe.solution
                if spe.mobile:
                    conc = conc * vol.material.get_solubility_coefficient(
                        self.mesh.mesh, self.temperature_fenics, spe
                    )
                velocity = drift_term.drift_velocity(
                    D=D, temperature=self.temperature_fenics
                )
                # assembled even when identically zero, as in the base class
                _drift.warn_if_no_effect(drift_term, spe, velocity)
                self.formulation += _drift.drift_form(
                    concentration=conc,
                    test_function=spe.test_function,
                    velocity=velocity,
                    dx=self.dx(vol.id),
                    coordinate_system=self.mesh.coordinate_system,
                    mesh=self.mesh.mesh,
                )

        for bc in self.boundary_conditions:
            if isinstance(bc, boundary_conditions.OutflowBC):
                volume = self.volume_subdomain_of_surface(bc.subdomain)
                velocity = self.drift_velocity_in(field=bc.species, volume=volume)
                if velocity is None:
                    continue
                conc = bc.species.solution
                if bc.species.mobile:
                    conc = conc * volume.material.get_solubility_coefficient(
                        self.mesh.mesh, self.temperature_fenics, bc.species
                    )
                self.formulation += (
                    conc
                    * ufl.dot(velocity, ufl.FacetNormal(self.mesh.mesh))
                    * bc.species.test_function
                    * self.ds(bc.subdomain.id)
                )

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
