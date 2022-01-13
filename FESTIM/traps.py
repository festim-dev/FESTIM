import sympy as sp
from fenics import *
from FESTIM import k_B, read_from_xdmf
import sympy as sp


class Concentration:
    """Class for concentrations (solute or traps) with attributed
    fenics.Function objects for the solution and the previous solution and a
    fenics.TestFunction

    Args:
        solution (fenics.Function or ufl.Indexed): Solution for "current"
            timestep
        previous_solution (fenics.Function or ufl.Indexed): Solution for "previous"
            timestep
        test_function (fenics.TestFunction or ufl.Indexed): test function
    """
    def __init__(self, solution=None, previous_solution=None, test_function=None):
        self.solution = solution
        self.previous_solution = previous_solution
        self.test_function = test_function
        self.sub_expressions = []
        self.F = None

    def initialise(self, initial_condition, V):
        if type(initial_condition['value']) == str and initial_condition['value'].endswith(".xdmf"):
            comp = read_from_xdmf(
                initial_condition["timestep"],
                initial_condition["label"],
                self.V)
        else:
            value = initial_condition["value"]
            value = sp.printing.ccode(value)
            comp = Expression(value, degree=3, t=0)
            comp = interpolate(comp, V)
        assign(self.previous_solution, comp)

    def read_from_xdmf(filename, timestep, label, V):
        comp = Function(V)
        with XDMFFile(ini["value"]) as f:
            f.read_checkpoint(comp, label, timestep)


class Mobile(Concentration):
    # TODO move this
    def __init__(self):
        super().__init__()

    def initialise(self, initial_condition, V, S=None):
        super().initialise(initial_condition, V)

        # variable change if chemical potential
        if S is not None:
            theta = self.previous_solution/S
            self.previous_solution.assign(project(theta, V))

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
                            Q * c_0 / (FESTIM.R * T**2) * grad(T),
                            grad(self.test_function))*dx(subdomain)

        # add the traps transient terms
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


class Trap(Concentration):
    def __init__(
            self, k_0, E_k, p_0, E_p, materials, density, source_term=0, id=None):
        super().__init__()
        self.id = id
        self.k_0 = k_0
        self.E_k = E_k
        self.p_0 = p_0
        self.E_p = E_p
        if type(materials) is not list:
            self.materials = [materials]
        else:
            self.materials = materials
        self.density = []
        self.make_density(density)
        self.source_term = source_term

    def make_density(self, densities):
        if type(densities) is not list:
            densities = [densities]

        for density in densities:
            density_expr = sp.printing.ccode(density)
            self.density.append(Expression(density_expr, degree=2, t=0))

    def create_form(
            self, mobile, materials, T, dx, dt=None,
            chemical_pot=False):
        """[summary]

        Args:
            mobile (FESTIM.Concentration): [description]
            materials (FESTIM.Materials): [description]
            T (FESTIM.Temperature): [description]
            dx ([type]): [description]
            dt ([type], optional): If None assuming steady state. Defaults to None.
            chemical_pot (bool, optional): [description]. Defaults to False.
        """
        self.F = 0
        self.create_trapping_form(mobile, materials, T, dx, dt, chemical_pot)

        self.create_source_form(dx)

    def create_trapping_form(self, mobile, materials, T, dx, dt=None, chemical_pot=False):
        solution = self.solution
        prev_solution = self.previous_solution
        test_function = self.test_function
        trap_materials = self.materials

        T = T.T
        c_0 = mobile.solution
        if chemical_pot:
            theta = c_0

        expressions_trap = []
        F_trapping = 0  # initialise the form

        if dt is not None:
            # d(c_t)/dt in trapping equation
            F_trapping += ((solution - prev_solution) / dt) * test_function * dx
        else:
            # if the sim is steady state and
            # if a trap is not defined in one subdomain
            # add c_t = 0 to the form in this subdomain
            all_mat_ids = [mat.id for mat in materials.materials]
            for mat_id in all_mat_ids:
                if mat_id not in trap_materials:
                    F_trapping += solution*test_function*dx(mat_id)

        for i, mat_id in enumerate(trap_materials):
            if type(self.k_0) is list:
                k_0 = self.k_0[i]
                E_k = self.E_k[i]
                p_0 = self.p_0[i]
                E_p = self.E_p[i]
                density = self.density[i]
            else:
                k_0 = self.k_0
                E_k = self.E_k
                p_0 = self.p_0
                E_p = self.E_p
                density = self.density[0]

            # add the density to the list of
            # expressions to be updated
            expressions_trap.append(density)

            corresponding_material = \
                materials.find_material_from_id(mat_id)
            if chemical_pot:
                # change of variable
                S_0 = corresponding_material.S_0
                E_S = corresponding_material.E_S
                c_0 = theta*S_0*exp(-E_S/k_B/T)

            # k(T)*c_m*(n - c_t) - p(T)*c_t
            F_trapping += - k_0 * exp(-E_k/k_B/T) * c_0 \
                * (density - solution) * \
                test_function*dx(mat_id)
            F_trapping += p_0*exp(-E_p/k_B/T)*solution * \
                test_function*dx(mat_id)

        self.F_trapping = F_trapping
        self.F += self.F_trapping
        self.sub_expressions += expressions_trap

    def create_source_form(self, dx):
        source = sp.printing.ccode(self.source_term)
        source = Expression(source, t=0, degree=2)
        self.F_source = -source*self.test_function*dx
        self.F += self.F_source
        self.sub_expressions.append(source)


class Traps:
    def __init__(self, traps=[]) -> None:
        self.traps = traps

        # add ids if unspecified
        for i, trap in enumerate(self.traps, 1):
            if trap.id is None:
                trap.id = i

    def create_forms(self, mobile, materials, T, dx, dt=None,
                     chemical_pot=False):
        for trap in self.traps:
            trap.create_form(mobile, materials, T, dx, dt=dt,
                             chemical_pot=chemical_pot)

    def get_trap(self, id):
        for trap in self.traps:
            print(trap.id)
            if trap.id == id:
                return trap
        raise ValueError("Couldn't find trap {}".format(id))
