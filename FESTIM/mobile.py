from FESTIM import Concentration, k_B, R
from fenics import *
import sympy as sp


class Mobile(Concentration):
    # TODO move this
    def __init__(self):
        super().__init__()

    def initialise(self, V, value, label=None, time_step=None, S=None):
        comp = self.get_comp(V, value, label=label, time_step=time_step)
        if S is None:
            comp = interpolate(comp, V)
        else:
            comp = comp/S
            # Product must be projected
            comp = project(comp, V)

        assign(self.previous_solution, comp)

    def create_form(self, materials, dx, T,  dt=None, traps=None, source_term=[], chemical_pot=False, soret=False):
        self.F = 0
        self.create_diffusion_form(materials, dx, T, dt=dt, traps=traps, chemical_pot=chemical_pot, soret=soret)
        self.create_source_form(dx, source_term)

    def create_diffusion_form(self, materials, dx, T, dt=None, traps=None, chemical_pot=False, soret=False):
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
                    F += ((c_0-c_0_n)/dt)*self.test_function*dx(subdomain)
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
                    F += ((trap.solution - trap.previous_solution) / dt) * \
                        self.test_function * dx
        self.F_diffusion = F
        self.F += F

    def create_source_form(self, dx, source_term):
        """[summary]

        Args:
            dx (fenics.Measure): [description]
            source_term (dict or list): {"value": ...} or [{"value": ..., "volumes": [1, 2]}, {"value": ..., "volumes": 3}]
        """
        F_source = 0
        expressions_source = []

        print('Defining source terms')

        if isinstance(source_term, dict):
            source = Expression(
                sp.printing.ccode(
                    source_term["value"]), t=0, degree=2)
            F_source += - source*self.test_function*dx
            expressions_source.append(source)

        elif isinstance(source_term, list):
            for source_dict in source_term:
                source = Expression(
                    sp.printing.ccode(
                        source_dict["value"]), t=0, degree=2)
                volumes = source_dict["volumes"]
                if isinstance(volumes, int):
                    volumes = [volumes]
                for vol in volumes:
                    print(vol)
                    F_source += - source*self.test_function*dx(vol)
                expressions_source.append(source)
        self.F_source = F_source
        self.F += F_source
        self.sub_expressions += expressions_source
