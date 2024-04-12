import festim as F
from dolfinx import fem
import ufl


class SurfaceReactionFlux(F.ParticleFluxBC):
    def __init__(
        self,
        reactant,
        gas_pressure,
        k_r0,
        E_kr,
        k_d0,
        E_kd,
        subdomain,
        species,
    ):
        self.reactant = reactant
        self.gas_pressure = gas_pressure
        self.k_r0 = k_r0
        self.E_kr = E_kr
        self.k_d0 = k_d0
        self.E_kd = E_kd
        super().__init__(subdomain=subdomain, value=None, species=species)

    def reaction_rate(self, temperature, t: fem.Constant):
        kr = self.k_r0 * ufl.exp(-self.E_kr / (F.k_B * temperature))
        kd = self.k_d0 * ufl.exp(-self.E_kd / (F.k_B * temperature))
        if callable(self.gas_pressure):
            gas_pressure = self.gas_pressure(t=t)
        else:
            gas_pressure = self.gas_pressure
        product_of_reactants = self.reactant[0].concentration
        for reactant in self.reactant[1:]:
            product_of_reactants *= reactant.concentration

        return kd * gas_pressure - kr * product_of_reactants

    def create_value_fenics(self, mesh, temperature, t: fem.Constant):
        self.value_fenics = self.reaction_rate(temperature, t)


class SurfaceReactionBC:
    def __init__(
        self,
        reactant,
        gas_pressure,
        k_r0,
        E_kr,
        k_d0,
        E_kd,
        subdomain,
    ):

        self.reactant = reactant
        self.gas_pressure = gas_pressure
        self.k_r0 = k_r0
        self.E_kr = E_kr
        self.k_d0 = k_d0
        self.E_kd = E_kd
        self.subdomain = subdomain

        self.flux_bcs = self.create_flux_bcs()

    @property
    def time_dependent(self):
        return False  # no need to update if only using ufl.conditional objects

    def create_flux_bcs(self):
        return [
            SurfaceReactionFlux(
                reactant=self.reactant,
                gas_pressure=self.gas_pressure,
                k_r0=self.k_r0,
                E_kr=self.E_kr,
                k_d0=self.k_d0,
                E_kd=self.E_kd,
                subdomain=self.subdomain,
                species=species,
            )
            for species in self.reactant
        ]
