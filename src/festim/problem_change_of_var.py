import festim.boundary_conditions
from festim.hydrogen_transport_problem import HydrogenTransportProblem
from festim.helpers import as_fenics_constant
from festim import boundary_conditions
import festim
import festim.species as _species

import ufl
from dolfinx import fem


class HydrogenTransportProblemDiscontinuousChangeVar(HydrogenTransportProblem):

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
                                not_defined_in_volume.remove(vol)

                    # add c = 0 to formulation where needed
                    for vol in not_defined_in_volume:
                        self.formulation += (
                            spe.solution * spe.test_function * self.dx(vol.id)
                        )

    # def define_boundary_conditions(self):
    #     for bc in self.boundary_conditions:
    #         if isinstance(bc.species, str):
    #             # if name of species is given then replace with species object
    #             bc.species = _species.find_species_from_name(bc.species, self.species)

    #     for bc in self.boundary_conditions:
    #         pass
