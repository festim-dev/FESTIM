from festim.boundary_conditions import ParticleFluxBC
from festim import k_B
from dolfinx import fem
import ufl


class SurfaceReactionBCpartial(ParticleFluxBC):
    """Boundary condition representing a surface reaction
    A + B <-> C
    where A, B are the reactants and C is the product
    the forward reaction rate is K_r = k_r0 * exp(-E_kr / (k_B * T))
    and the backward reaction rate is K_d = k_d0 * exp(-E_kd / (k_B * T))
    The reaction rate is:
    K = K_r * C_A * C_B - K_d * P_C
    with C_A, C_B the concentration of species A and B,
    P_C the partial pressure of species C at the surface.

    This class is used to create the flux of a single species entering the surface
    Example: The flux of species A entering the surface is K.


    Args:
        reactant (list): list of F.Species objects representing the reactants
        gas_pressure (float, callable): the partial pressure of the product species
        k_r0 (float): the pre-exponential factor of the forward reaction rate
        E_kr (float): the activation energy of the forward reaction rate (eV)
        k_d0 (float): the pre-exponential factor of the backward reaction rate
        E_kd (float): the activation energy of the backward reaction rate (eV)
        subdomain (F.SurfaceSubdomain): the surface subdomain where the reaction occurs
        species (F.Species): the species to which the flux is applied
    """

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

    def create_value_fenics(self, mesh, temperature, t: fem.Constant):
        kr = self.k_r0 * ufl.exp(-self.E_kr / (k_B * temperature))
        kd = self.k_d0 * ufl.exp(-self.E_kd / (k_B * temperature))
        if callable(self.gas_pressure):
            gas_pressure = self.gas_pressure(t=t)
        else:
            gas_pressure = self.gas_pressure

        product_of_reactants = self.reactant[0].concentration
        for reactant in self.reactant[1:]:
            product_of_reactants *= reactant.concentration

        self.value_fenics = kd * gas_pressure - kr * product_of_reactants


class SurfaceReactionBC:
    """Boundary condition representing a surface reaction
    A + B <-> C
    where A, B are the reactants and C is the product
    the forward reaction rate is K_r = k_r0 * exp(-E_kr / (k_B * T))
    and the backward reaction rate is K_d = k_d0 * exp(-E_kd / (k_B * T))
    The reaction rate is:
    K = K_r * C_A * C_B - K_d * P_C
    with C_A, C_B the concentration of species A and B,
    P_C the partial pressure of species C at the surface.

    The flux of species A entering the surface is K.
    In the special case where A=B, then the flux of particle entering the surface is 2*K


    Args:
        reactant (list): list of F.Species objects representing the reactants
        gas_pressure (float, callable): the partial pressure of the product species
        k_r0 (float): the pre-exponential factor of the forward reaction rate
        E_kr (float): the activation energy of the forward reaction rate (eV)
        k_d0 (float): the pre-exponential factor of the backward reaction rate
        E_kd (float): the activation energy of the backward reaction rate (eV)
        subdomain (F.SurfaceSubdomain): the surface subdomain where the reaction occurs
    """

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

        # create the flux boundary condition for each reactant
        self.flux_bcs = [
            SurfaceReactionBCpartial(
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

    @property
    def time_dependent(self):
        return False  # no need to update if only using ufl.conditional objects
