import festim.boundary_conditions
from festim.hydrogen_transport_problem import HydrogenTransportProblem
from festim import boundary_conditions, as_fenics_constant
import festim
import festim.species as _species
import festim.exports as exports
import ufl
from dolfinx import fem

from typing import List


class HydrogenTransportProblemDiscontinuousChangeVar(HydrogenTransportProblem):
    species: List[_species.Species]

    def initialise(self):
        self.create_species_from_traps()
        self.define_function_spaces()
        self.define_meshtags_and_measures()
        self.assign_functions_to_species()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self.dt = as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )

        self.define_temperature()
        self.define_boundary_conditions()
        self.create_source_values_fenics()
        self.create_flux_values_fenics()
        self.create_initial_conditions()
        self.create_formulation()
        self.create_solver()
        self.override_post_processing_solution()
        self.initialise_exports()

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
                K_S = vol.material.get_solubility_coefficient(
                    self.mesh.mesh, self.temperature_fenics, spe
                )
                if spe.mobile:
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
            if isinstance(reaction.product, list):
                products = reaction.product
            else:
                products = [reaction.product]

            # hack enforce the concentration attribute of the species for all species
            # to be used in reaction.reaction_term
            for spe in self.species:
                K_S = reaction.volume.material.get_solubility_coefficient(
                    self.mesh.mesh, self.temperature_fenics, spe
                )
                if spe.mobile:
                    spe.concentration = spe.solution * K_S

            # reactant
            for reactant in reaction.reactant:
                if isinstance(reactant, festim.species.Species):
                    self.formulation += (
                        reaction.reaction_term(self.temperature_fenics)
                        * reactant.test_function
                        * self.dx(reaction.volume.id)
                    )

            # product
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
                            if vol == reaction.volume:
                                if vol in not_defined_in_volume:
                                    not_defined_in_volume.remove(vol)

                    # add c = 0 to formulation where needed
                    for vol in not_defined_in_volume:
                        self.formulation += (
                            spe.solution * spe.test_function * self.dx(vol.id)
                        )

    def override_post_processing_solution(self):
        # override the post-processing solution c = theta * K_S
        Q0 = fem.functionspace(self.mesh.mesh, ("DG", 0))
        Q1 = fem.functionspace(self.mesh.mesh, ("DG", 1))
        K_S0 = fem.Function(Q0)
        E_KS = fem.Function(Q0)
        for subdomain in self.volume_subdomains:
            entities = subdomain.locate_subdomain_entities(self.mesh.mesh)
            K_S0.x.array[entities] = subdomain.material.K_S_0
            E_KS.x.array[entities] = subdomain.material.E_K_S

        K_S = K_S0 * ufl.exp(-E_KS / (festim.k_B * self.temperature_fenics))

        for spe in self.species:
            theta = spe.solution

            spe.dg_expr = fem.Expression(theta * K_S, Q1.element.interpolation_points())
            spe.post_processing_solution = fem.Function(Q1)
            spe.post_processing_solution.interpolate(spe.dg_expr)

    def post_processing(self):

        # need to compute c = theta * K_S
        # this expression is stored in species.dg_expr
        for spe in self.species:
            spe.post_processing_solution.interpolate(spe.dg_expr)

        super().post_processing()

    # def define_boundary_conditions(self):
    #     for bc in self.boundary_conditions:
    #         if isinstance(bc.species, str):
    #             # if name of species is given then replace with species object
    #             bc.species = _species.find_species_from_name(bc.species, self.species)

    #     for bc in self.boundary_conditions:
    #         pass
