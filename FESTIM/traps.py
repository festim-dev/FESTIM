import sympy as sp
from fenics import *
from FESTIM import R, k_B, read_from_xdmf
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
        self.post_processing_solution = None  # used for post treatment

    def initialise(self, initial_condition, V):
        comp = self.get_comp(initial_condition, V)
        comp = interpolate(comp, V)
        assign(self.previous_solution, comp)

    def get_comp(self, initial_condition, V):
        if type(initial_condition['value']) == str and initial_condition['value'].endswith(".xdmf"):
            comp = read_from_xdmf(
                initial_condition['value'],
                initial_condition["label"],
                initial_condition["time_step"],
                V)
        else:
            value = initial_condition["value"]
            value = sp.printing.ccode(value)
            comp = Expression(value, degree=3, t=0)
        return comp

    def read_from_xdmf(filename, timestep, label, V):
        comp = Function(V)
        with XDMFFile(ini["value"]) as f:
            f.read_checkpoint(comp, label, timestep)


class Mobile(Concentration):
    # TODO move this
    def __init__(self):
        super().__init__()

    def initialise(self, initial_condition, V, S=None):
        comp = self.get_comp(initial_condition, V)

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


class Trap(Concentration):
    def __init__(
            self, k_0, E_k, p_0, E_p, materials, density, source_term=None, id=None):
        super().__init__()
        self.id = id
        self.k_0 = k_0
        self.E_k = E_k
        self.p_0 = p_0
        self.E_p = E_p
        self.materials = materials
        if not isinstance(self.materials, list):
            self.materials = [self.materials]
        self.density = []
        self.make_density(density)
        self.source_term = source_term

    def make_density(self, densities):
        if type(densities) is not list:
            densities = [densities]

        for density in densities:
            if density is not None:
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
        if self.source_term is not None:
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
        self.F = None
        self.sub_expressions = []

        # add ids if unspecified
        for i, trap in enumerate(self.traps, 1):
            if trap.id is None:
                trap.id = i

    def create_forms(self, mobile, materials, T, dx, dt=None,
                     chemical_pot=False):
        self.F = 0
        for trap in self.traps:
            trap.create_form(mobile, materials, T, dx, dt=dt,
                             chemical_pot=chemical_pot)
            self.F += trap.F
            self.sub_expressions += trap.sub_expressions

    def get_trap(self, id):
        for trap in self.traps:
            if trap.id == id:
                return trap
        raise ValueError("Couldn't find trap {}".format(id))


class ExtrinsicTrap(Trap):
    def __init__(self, k_0, E_k, p_0, E_p, materials, form_parameters, id=None, type=None):
        super().__init__(k_0, E_k, p_0, E_p, materials, density=None, id=id)
        self.form_parameters = form_parameters
        self.density_previous_solution = None
        self.density_test_function = None
        self.type = type

    def convert_prms(self):
        # create Expressions or Constant for all parameters
        for key, value in self.form_parameters.items():
            if isinstance(value, (int, float)):
                self.prms[key] = Constant(value)
            else:
                self.prms[key] = Expression(sp.printing.ccode(value),
                                       t=0,
                                       degree=1)
                self.sub_expressions.append(self.prms[key])

    def create_form_density(self, dx, dt):
        phi_0 = self.form_parameters["phi_0"]
        n_amax = self.form_parameters["n_amax"]
        n_bmax = self.form_parameters["n_bmax"]
        eta_a = self.form_parameters["eta_a"]
        eta_b = self.form_parameters["eta_b"]
        f_a = self.form_parameters["f_a"]
        f_b = self.form_parameters["f_b"]
        density = self.density[0]
        F = ((density - self.density_previous_solution)/dt) * \
            self.density_test_function*dx
        F += -phi_0*(
            (1 - density/n_amax)*eta_a*f_a +
            (1 - density/n_bmax)*eta_b*f_b) \
            * self.density_test_function*dx
        self.form_density = F
