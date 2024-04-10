import festim as F
import fenics as f


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

        value = self.reaction_rate()

        super().__init__(subdomain, value, species)

    def reaction_rate(self, T):
        kr = self.k_r0 * f.exp(-self.E_kr / (F.k_B * T))
        kd = self.k_d0 * f.exp(-self.E_kd / (F.k_B * T))
        if callable(
            self.gas_pressure
        ):  # need to check if gas_pressure is a function of time only
            return lambda t: kd * self.gas_pressure(t) - kr * self.reactant
        else:
            return kd * self.gas_pressure - kr * self.reactant
