from FESTIM import Concentration, k_B, R
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

    def initialise(self, V, value, label=None, time_step=None, S=None):
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
        if S is None:
            comp = interpolate(comp, V)
        else:
            comp = comp/S
            # Product must be projected
            comp = project(comp, V)

        assign(self.previous_solution, comp)

    def create_form(self, materials, dx, T,  dt=None, traps=None, chemical_pot=False, soret=False):
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
        self.create_diffusion_form(materials, dx, T, dt=dt, traps=traps, chemical_pot=chemical_pot, soret=soret)
        self.create_source_form(dx)

    def create_diffusion_form(self, materials, dx, T, dt=None, traps=None, chemical_pot=False, soret=False):
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
        c_0 = self.solution
        c_0_n = self.previous_solution
        if chemical_pot:
            theta = c_0
            theta_n = c_0_n
        T, T_n = T.T, T.T_n

        for material in materials.materials:
            D_0 = material.D_0
            E_D = material.E_D
            if chemical_pot:
                E_S = material.E_S
                S_0 = material.S_0
                c_0 = theta*S_0*exp(-E_S/k_B/T)
                c_0_n = theta_n*S_0*exp(-E_S/k_B/T_n)

            subdomains = material.id  # list of subdomains with this material
            if type(subdomains) is not list:
                subdomains = [subdomains]  # make sure subdomains is a list

            # add to the formulation F for every subdomain
            for subdomain in subdomains:
                # transient form
                if dt is not None:
                    F += ((c_0-c_0_n)/dt.value)*self.test_function*dx(subdomain)
                F += dot(D_0 * exp(-E_D/k_B/T)*grad(c_0),
                        grad(self.test_function))*dx(subdomain)
                if soret:
                    Q = material.free_enthalpy*T + material.entropy
                    F += dot(D_0 * exp(-E_D/k_B/T) *
                            Q * c_0 / (R * T**2) * grad(T),
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
