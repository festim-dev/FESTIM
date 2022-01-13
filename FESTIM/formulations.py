from fenics import split, grad, dot, Expression, exp
import sympy as sp
import FESTIM


class Trapold(FESTIM.Concentration):
    """Class for traps inheriting from Concentration() which has usefull
    additional attributes (k_0, E_k, p_0, E_p, density, type, materials)

        Args:
            trap_dict (dict): contains the trap properties. Ex:
                {
                    "k_0": 1,
                    "E_k": 2,
                    "p_0": 3,
                    "E_p": 4,
                    "density": 5,
                    "materials": [1, 2]
                }
            simulation (FESTIM.Simulation): main simulation instance
            extrinsic_counter (int): counter for extrinsic traps usefull to
                attribute the correct function for trap density
    """
    def __init__(
            self, trap_dict, simulation, extrinsic_counter, **kwargs):
        super().__init__(**kwargs)

        self.k_0 = trap_dict["k_0"]
        self.E_k = trap_dict["E_k"]
        self.p_0 = trap_dict["p_0"]
        self.E_p = trap_dict["E_p"]
        self.density = []

        if 'type' in trap_dict and trap_dict['type'] == 'extrinsic':
            self.type = "extrinsic"
            density = simulation.extrinsic_traps[extrinsic_counter]
            self.density.append(density)
        else:
            # make sure .density is a list
            if type(trap_dict['density']) is not list:
                densities = [trap_dict['density']]
            else:
                densities = trap_dict['density']

                # check that there are no duplicate ids
                if len(set(trap_dict['materials'])) != len(trap_dict['materials']):
                    msg = "The key 'density' contains duplicated " + \
                        "ids"
                    raise ValueError(msg)
            for density in densities:
                density_expr = sp.printing.ccode(density)
                self.density.append(Expression(density_expr, degree=2, t=0))
        if type(trap_dict['materials']) is not list:
            self.materials = [trap_dict['materials']]
        else:
            self.materials = trap_dict['materials']


def formulation(simulation):
    """Creates the variational formulation for the H transport problem

    Args:
        simulation (FESTIM.Simulation): main simulation instance

    Returns:
        fenics.Form, list: problem variational formulation, contains
            fenics.Expression() to be updated
    """
    expressions = []
    F = 0

    # diffusion + transient terms
    if "source_term" in simulation.parameters:
        source_term = simulation.parameters["source_term"]
    else:
        source_term = []
    simulation.mobile.create_form(
        simulation.materials, simulation.dx, simulation.T, simulation.dt,
        traps=simulation.traps, source_term=source_term, chemical_pot=simulation.chemical_pot,
        soret=simulation.soret)
    F += simulation.mobile.F
    expressions += simulation.mobile.sub_expressions

    # Add traps
    if "traps" in simulation.parameters:
        F_traps, expressions_traps = \
            create_all_traps_form(simulation, simulation.mobile.solution, simulation.mobile.test_function)
        F += F_traps
        expressions += expressions_traps

    return F, expressions


def create_one_trap_form(simulation, trap, c_0, v):
    """Creates a sub-form for a trap to be added to the general formulation.

    The global equation for trapping is:
    d(c_t)/dt = k_0*exp(-E_k/k_B T) * c_m * (n - c _t)
    - p_0*exp(-E_p/k_B T)*c_t

    Args:
        simulation (FESTIM.Simulation): the main simulation object
        trap (Trap): an instance of the Trap() class
        solute (Concentration): an instance of the Concentration() class for
            the solute concentration

    Returns:
        fenics.Form, list: the form related to the trap, list of
            sources as fenics.Expression
    """
    k_B = FESTIM.k_B  # Boltzmann constant

    solution = trap.solution
    prev_solution = trap.prev_solution
    test_function = trap.test_function
    trap_materials = trap.materials

    materials = simulation.materials
    dt = simulation.dt
    dx = simulation.dx
    T = simulation.T.T

    if simulation.chemical_pot:
        theta = c_0

    expressions_trap = []
    F = 0  # initialise the form
    if simulation.transient:
        # d(c_t)/dt in trapping equation
        F += ((solution - prev_solution) / dt) * \
            test_function*dx
        # d(c_t)/dt in mobile equation
        F += ((solution - prev_solution) / dt) * \
            v*dx
    else:
        # if the sim is steady state and
        # if a trap is not defined in one subdomain
        # add c_t = 0 to the form in this subdomain
        all_mat_ids = [mat.id for mat in materials.materials]
        for mat_id in all_mat_ids:
            if mat_id not in trap_materials:
                F += solution*test_function*dx(mat_id)

    for i, mat_id in enumerate(trap_materials):
        if type(trap.k_0) is list:
            k_0 = trap.k_0[i]
            E_k = trap.E_k[i]
            p_0 = trap.p_0[i]
            E_p = trap.E_p[i]
            density = trap.density[i]
        else:
            k_0 = trap.k_0
            E_k = trap.E_k
            p_0 = trap.p_0
            E_p = trap.E_p
            density = trap.density[0]

        # add the density to the list of
        # expressions to be updated
        expressions_trap.append(density)

        corresponding_material = \
            simulation.materials.find_material_from_id(mat_id)
        if simulation.chemical_pot:
            # change of variable
            S_0 = corresponding_material.S_0
            E_S = corresponding_material.E_S
            c_0 = theta*S_0*exp(-E_S/k_B/T)

        # k(T)*c_m*(n - c_t) - p(T)*c_t
        F += - k_0 * exp(-E_k/k_B/T) * c_0 \
            * (density - solution) * \
            test_function*dx(mat_id)
        F += p_0*exp(-E_p/k_B/T)*solution * \
            test_function*dx(mat_id)
    return F, expressions_trap


def create_all_traps_form(simulation, c_0, v):
    """Creates a sub-form for all traps to be added to the general formulation.

    Args:
        simulation (FESTIM.Simulation): the main simulation object
        solute (Concentration): an instance of the Concentration() class for
            the solute concentration

    Returns:
        fenics.Form, list: formulation for the trapping terms, list containing
            densities and sources as fenics.Expression
    """
    F_traps = 0
    expressions_traps = []

    parameters = simulation.parameters
    solutions = split(simulation.u)
    previous_solutions = split(simulation.u_n)
    testfunctions = split(simulation.v)

    extrinsic_counter = 0  # index for extrinsic_traps
    for i, trap_dict in enumerate(parameters["traps"], 1):

        trap_object = Trap(
            trap_dict, simulation, extrinsic_counter, solution=solutions[i],
            prev_solution=previous_solutions[i],
            test_function=testfunctions[i])

        # increment extrinsic_counter
        if hasattr(trap_object, "type"):
            extrinsic_counter += 1

        # add to the global form
        F_trap, expressions_trap = create_one_trap_form(
            simulation, trap_object, c_0, v)
        F_traps += F_trap
        expressions_traps += expressions_trap

        # if a source term is set then add it to the form
        if 'source_term' in trap_dict:
            source = sp.printing.ccode(trap_dict['source_term'])
            source = Expression(source, t=0, degree=2)
            F_traps += -source*testfunctions[i]*simulation.dx
            expressions_traps.append(source)
    return F_traps, expressions_traps


def formulation_extrinsic_traps(simulation):
    """Creates a list that contains formulations to be solved during
    time stepping.

    Arguments:


    Returns:
        list -- contains fenics.Form to be solved for extrinsic trap density
        list -- contains fenics.Expression to be updated
    """
    traps = simulation.parameters["traps"]
    solutions = simulation.extrinsic_traps
    previous_solutions = simulation.previous_solutions_traps
    testfunctions = simulation.testfunctions_traps
    dt = simulation.dt
    dx = simulation.dx

    formulations = []
    expressions = []
    i = 0
    for trap in traps:
        if 'type' in trap.keys():
            if trap['type'] == 'extrinsic':
                parameters = trap["form_parameters"]
                phi_0 = sp.printing.ccode(parameters['phi_0'])
                phi_0 = Expression(phi_0, t=0, degree=2)
                expressions.append(phi_0)
                n_amax = parameters['n_amax']
                n_bmax = parameters['n_bmax']
                eta_a = parameters['eta_a']
                eta_b = parameters['eta_b']
                f_a = sp.printing.ccode(parameters['f_a'])
                f_a = Expression(f_a, t=0, degree=2)
                expressions.append(f_a)
                f_b = sp.printing.ccode(parameters['f_b'])
                f_b = Expression(f_b, t=0, degree=2)
                expressions.append(f_b)

                F = ((solutions[i] - previous_solutions[i])/dt) * \
                    testfunctions[i]*dx
                F += -phi_0*(
                    (1 - solutions[i]/n_amax)*eta_a*f_a +
                    (1 - solutions[i]/n_bmax)*eta_b*f_b) \
                    * testfunctions[i]*dx
                formulations.append(F)
                i += 1
    return formulations, expressions
