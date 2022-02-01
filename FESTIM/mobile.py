from FESTIM import Concentration, FluxBC, k_B, R
from fenics import *
import sympy as sp


class Mobile(Concentration):
    """
    The mobile concentration.

    If conservation of chemical potential, this will be c_m/S.
    If not, Mobile represents c_m.

    Attributes:
        sources (list): list of FESTIM.Source objects.
            The volumetric source terms
        F (fenics.Form): the variational formulation for mobile
    """
    def __init__(self):
        """Inits FESTIM.Mobile
        """
        super().__init__()
        self.sources = []
        self.boundary_conditions = []

    def initialise(self, V, value, label=None, time_step=None):
        """Assign a value to self.previous_solution

        Args:
            V (fenics.FunctionSpace): the function space
            value (sp.Add, float, int, str): the value of the initialisation.
            label (str, optional): the label in the XDMF file. Defaults to
                None.
            time_step (int, optional): the time step to read in the XDMF file.
                Defaults to None.
            S (FESTIM.ArheniusCoeff, optional): the solubility. If not None,
                conservation of chemical potential is assumed. Defaults to
                None.
        """
        comp = self.get_comp(V, value, label=label, time_step=time_step)
        comp = interpolate(comp, V)

        assign(self.previous_solution, comp)

    def create_form(self, materials, dx, ds, T,  dt=None, traps=None, soret=False):
        """Creates the variational formulation.

        Args:
            materials (FESTIM.Materials): the materials
            dx (fenics.Measure): the measure dx
            T (FESTIM.Temperature): the temperature
            dt (FESTIM.Stepsize, optional): the stepsize. Defaults to None.
            traps (FESTIM.Traps, optional): the traps. Defaults to None.
            chemical_pot (bool, optional): if True, conservation of chemical
                potential is assumed. Defaults to False.
            soret (bool, optional): If True, Soret effect is assumed. Defaults
                to False.
        """
        self.F = 0
        self.create_diffusion_form(materials, dx, T, dt=dt, traps=traps, soret=soret)
        self.create_source_form(dx)
        self.create_fluxes_form(materials, T, ds)

    def create_diffusion_form(self, materials, dx, T, dt=None, traps=None, soret=False):
        """Creates the variational formulation for the diffusive part.

        Args:
            materials (FESTIM.Materials): the materials
            dx (fenics.Measure): the measure dx
            T (FESTIM.Temperature): the temperature
            dt (FESTIM.Stepsize, optional): the stepsize. Defaults to None.
            traps (FESTIM.Traps, optional): the traps. Defaults to None.
            chemical_pot (bool, optional): if True, conservation of chemical
                potential is assumed. Defaults to False.
            soret (bool, optional): If True, Soret effect is assumed. Defaults
                to False.
        """
        F = 0
        for material in materials.materials:
            D_0 = material.D_0
            E_D = material.E_D
            c_0, c_0_n = self.get_concentration_for_a_given_material(material, T)

            subdomains = material.id  # list of subdomains with this material
            if type(subdomains) is not list:
                subdomains = [subdomains]  # make sure subdomains is a list

            # add to the formulation F for every subdomain
            for subdomain in subdomains:
                # transient form
                if dt is not None:
                    F += ((c_0-c_0_n)/dt.value)*self.test_function*dx(subdomain)
                F += dot(D_0 * exp(-E_D/k_B/T.T)*grad(c_0),
                         grad(self.test_function))*dx(subdomain)
                if soret:
                    Q = material.free_enthalpy*T.T + material.entropy
                    F += dot(D_0 * exp(-E_D/k_B/T.T) *
                             Q * c_0 / (R * T.T**2) * grad(T.T),
                             grad(self.test_function))*dx(subdomain)

        # add the traps transient terms
        if dt is not None:
            if traps is not None:
                for trap in traps.traps:
                    F += ((trap.solution - trap.previous_solution) / dt.value) * \
                        self.test_function * dx
        self.F_diffusion = F
        self.F += F

    def create_source_form(self, dx):
        """Creates the variational form for the volumetric source term parts.

        Args:
            dx (fenics.Measure): the measure dx
        """
        F_source = 0
        expressions_source = []

        print('Defining source terms')
        for source_term in self.sources:
            source = Expression(
                sp.printing.ccode(
                    source_term.value), t=0, degree=2)
            if type(source_term.volume) is list:
                volumes = source_term.volume
            else:
                volumes = [source_term.volume]
            for volume in volumes:
                F_source += - source*self.test_function*dx(volume)
            expressions_source.append(source)

        self.F_source = F_source
        self.F += F_source
        self.sub_expressions += expressions_source

    def create_fluxes_form(self, materials, T, ds):
        """Modifies the formulation and adds fluxes based
        on parameters in self.boundary_conditions
        """

        expressions_fluxes = []
        F = 0

        solute = self.get_solute_concentration(materials)

        for bc in self.boundary_conditions:
            if bc.component != "T":
                if isinstance(bc, FluxBC):
                    bc.create_form(T.T, solute)
                    # TODO : one day we will get rid of this huge expressions list
                    expressions_fluxes += bc.sub_expressions

                    for surf in bc.surfaces:
                        F += -self.test_function*bc.form*ds(surf)
        self.F_fluxes = F
        self.F += F
        self.sub_expressions += expressions_fluxes

    def get_concentration_for_a_given_material(self, material, T):
        return self.solution, self.previous_solution

    def get_solute_concentration(self, materials):
        return self.solution
