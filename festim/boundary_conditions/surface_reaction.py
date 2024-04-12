import festim as F
import ufl


# TODO instead of inheriting from ParticleFluxBC, it should be composed of several ParticleFluxBCs
class SurfaceReactionBC(F.ParticleFluxBC):
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
        name_to_species = {f"c{i}": species for i, species in enumerate(self.reactant)}

        value = self.create_value()

        super().__init__(
            subdomain, value, species, species_dependent_value=name_to_species
        )

    def create_value(self):
        kwargs = {f"c{i}": None for i, _ in enumerate(self.reactant)}
        kwargs["T"] = None
        if callable(self.gas_pressure):
            kwargs["t"] = None

        def value(**kwargs):
            T = kwargs["T"]
            kr = self.k_r0 * ufl.exp(-self.E_kr / (F.k_B * T))
            kd = self.k_d0 * ufl.exp(-self.E_kd / (F.k_B * T))
            if callable(self.gas_pressure):
                gas_pressure = self.gas_pressure(t=kwargs["t"])
            else:
                gas_pressure = self.gas_pressure
            return kd * gas_pressure - kr * self.reactant

        return value
