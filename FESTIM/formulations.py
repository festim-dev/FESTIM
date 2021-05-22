from fenics import split, grad, dot, Expression, exp, dx
import sympy as sp
import FESTIM


def formulation(simulation):
    """Creates formulation for trapping MRE model

    Arguments:


    Returns:
        fenics.Form() -- global formulation
        list -- contains fenics.Expression() to be updated
    """
    u, u_n = simulation.u, simulation.u_n
    v = simulation.v
    parameters = simulation.parameters
    T, T_n = simulation.T, simulation.T_n
    dt = simulation.dt
    dx = simulation.dx
    k_B = FESTIM.k_B  # Boltzmann constant
    expressions = []
    F = 0

    chemical_pot = False
    soret = False
    if "temperature" in parameters:
        if "soret" in parameters["temperature"]:
            if parameters["temperature"]["soret"] is True:
                soret = True
    solutions = split(u)
    previous_solutions = split(u_n)
    testfunctions = split(v)
    c_0 = solutions[0]
    c_0_n = previous_solutions[0]

    for material in parameters["materials"]:
        D_0 = material['D_0']
        E_D = material['E_D']
        if "S_0" in material or "E_S" in material:
            chemical_pot = True
            E_S = material['E_S']
            S_0 = material['S_0']
            c_0 = solutions[0]*S_0*exp(-E_S/k_B/T)
            c_0_n = previous_solutions[0]*S_0*exp(-E_S/k_B/T_n)

        subdomain = material['id']
        if simulation.transient:
            F += ((c_0-c_0_n)/dt)*testfunctions[0]*dx(subdomain)
        F += dot(D_0 * exp(-E_D/k_B/T)*grad(c_0),
                 grad(testfunctions[0]))*dx(subdomain)
        if soret:
            Q = material["H"]["free_enthalpy"]*T + material["H"]["entropy"]
            F += dot(D_0 * exp(-E_D/k_B/T) *
                     Q * c_0 / (FESTIM.R * T**2) * grad(T),
                     grad(testfunctions[0]))*dx(subdomain)
    # Define flux
    if "source_term" in parameters:
        print('Defining source terms')
        if isinstance(parameters["source_term"], dict):
            source = Expression(
                sp.printing.ccode(
                    parameters["source_term"]["value"]), t=0, degree=2)
            F += - source*testfunctions[0]*dx
            expressions.append(source)
        elif isinstance(parameters["source_term"], list):
            for source_dict in parameters["source_term"]:
                source = Expression(
                    sp.printing.ccode(
                        source_dict["value"]), t=0, degree=2)
                volumes = source_dict["volumes"]
                if isinstance(volumes, int):
                    volumes = [volumes]
                for vol in volumes:
                    F += - source*testfunctions[0]*dx(vol)
                expressions.append(source)
    expressions.append(T)  # Add it to the expressions to be updated

    # Add traps
    solute_object = Concentration(
        solutions[0], previous_solutions[0], testfunctions[0])

    extrinsic_counter = 0  # index in extrinsic_traps
    for i, trap_dict in enumerate(parameters["traps"]):

        trap_object = Trap(
            trap_dict, simulation, extrinsic_counter, solution=solutions[i],
            prev_solution=previous_solutions[i],
            test_function=testfunctions[i])

        # increment extrinsic_counter
        if hasattr(trap_object, "type"):
            extrinsic_counter += 1
        expressions.append(trap_object.density)

        # add to the global form
        F += create_trap_form(
            trap_object, solute_object, T,
            dt, dx, simulation.transient, chemical_pot,
            parameters["materials"])

        # if a source term is set then add it to the form
        if 'source_term' in trap_dict:
            source = sp.printing.ccode(trap_dict['source_term'])
            source = Expression(source, t=0, degree=2)
            F += -source*testfunctions[i]*dx
            expressions.append(source)

    return F, expressions


class Concentration:
    def __init__(self, solution, prev_solution, test_function):
        self.solution = solution
        self.prev_solution = prev_solution
        self.test_function = test_function


class Trap(Concentration):
    def __init__(
            self, trap_dict, simulation, extrinsic_counter, **kwargs):
        super().__init__(**kwargs)

        self.k_0 = trap_dict["k_0"]
        self.E_k = trap_dict["E_k"]
        self.p_0 = trap_dict["p_0"]
        self.E_p = trap_dict["E_p"]

        if 'type' in trap_dict and trap_dict['type'] == 'extrinsic':
            self.type = "extrinsic"
            density = simulation.extrinsic_traps[extrinsic_counter]
        else:
            density = sp.printing.ccode(trap_dict['density'])
            density = Expression(density, degree=2, t=0)
        self.density = density
        if type(trap_dict['materials']) is not list:
            self.materials = [trap_dict['materials']]
        else:
            self.materials = trap_dict['materials']


def create_trap_form(
        trap, solute, T, dt, dx, transient,
        chemical_pot, materials):
    """Creates a sub-form for a trap to be added to the general formulation.

    The global equation for trapping is:
    d(c_t)/dt = k_0*exp(-E_k/k_B T) * c_m * (n - c _t)
    - p_0*exp(-E_p/k_B T)*c_t

    Args:
        trap (Trap): an instance of the Trap() class
        solute (Concentration): an instance of the Concentration() class for
            the solute concentration
        T (fenics.Function): the temperature
        dt (fenics.Constant): the stepsize
        dx (fenics.Measure): the measure dx
        transient (bool): If True, transient terms (dc/dt) will be added.
        chemical_pot (bool): If True, the variable change
            solute = solute * S(T) will be made
        materials (list): Contains dicts. Ex:
            [
                {
                    "S_0": 0.2  # only needed if chemical_pot is True
                    "E_S": 0.5  # only needed if chemical_pot is True
                    "id": 1
                }
            ]

    Returns:
        fenics.Form: the form related to the trap
    """
    k_B = FESTIM.k_B  # Boltzmann constant

    solution = trap.solution
    prev_solution = trap.prev_solution
    test_function = trap.test_function
    trap_materials = trap.materials

    F = 0  # initialise the form
    if transient:
        # d(c_t)/dt in trapping equation
        F += ((solution - prev_solution) / dt) * \
            test_function*dx
        # d(c_t)/dt in mobile equation
        F += ((solution - prev_solution) / dt) * \
            solute.test_function*dx
    else:
        # if the sim is steady state and
        # if a trap is not defined in one subdomain
        # add c_t = 0 to the form in this subdomain
        all_mat_ids = [mat["id"] for mat in materials]
        for mat_id in all_mat_ids:
            if mat_id not in trap_materials:
                F += solution*test_function*dx(mat_id)

    for mat_id in trap_materials:
        k_0 = trap.k_0
        E_k = trap.E_k
        p_0 = trap.p_0
        E_p = trap.E_p
        density = trap.density
        corresponding_material = \
            FESTIM.helpers.find_material_from_id(
                materials, mat_id)
        c_0 = solute.solution
        if chemical_pot is True:
            # change of variable
            S_0 = corresponding_material['S_0']
            E_S = corresponding_material['E_S']
            c_0 = c_0*S_0*exp(-E_S/k_B/T)

        # k(T)*c_m*(n - c_t) - p(T)*c_t
        F += - k_0 * exp(-E_k/k_B/T) * c_0 \
            * (density - solution) * \
            test_function*dx(mat_id)
        F += p_0*exp(-E_p/k_B/T)*solution * \
            test_function*dx(mat_id)

    return F


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


def define_variational_problem_heat_transfers(simulation):
    """Create a variational form for heat transfer problem

    Arguments:

    Raises:
        NameError: if thermal_cond is not in keys
        NameError: if heat_capacity is not in keys
        NameError: if rho is not in keys

    Returns:
        fenics.Form -- the formulation for heat transfers problem
        list -- contains the fenics.Expression to be updated
    """

    print('Defining variational problem heat transfers')
    expressions = []
    parameters = simulation.parameters
    dx = simulation.dx
    ds = simulation.ds
    dt = simulation.dt
    T, T_n = simulation.T, simulation.T_n
    vT = simulation.vT

    F = 0
    for mat in parameters["materials"]:
        if "thermal_cond" not in mat.keys():
            raise NameError("Missing thermal_cond key in material")
        thermal_cond = mat["thermal_cond"]
        if callable(thermal_cond):  # if thermal_cond is a function
            thermal_cond = thermal_cond(T)
        vol = mat["id"]
        if parameters["temperature"]["type"] == "solve_transient":
            if "heat_capacity" not in mat.keys():
                raise NameError("Missing heat_capacity key in material")
            if "rho" not in mat.keys():
                raise NameError("Missing rho key in material")
            cp = mat["heat_capacity"]
            rho = mat["rho"]
            if callable(cp):  # if cp or rho are functions, apply T
                cp = cp(T)
            if callable(rho):
                rho = rho(T)
            # Transien term
            F += rho*cp*(T-T_n)/dt*vT*dx(vol)
        # Diffusion term
        F += dot(thermal_cond*grad(T), grad(vT))*dx(vol)

    # Source terms
    if "source_term" in parameters["temperature"].keys():
        for source in parameters["temperature"]["source_term"]:
            src = sp.printing.ccode(source["value"])
            src = Expression(src, degree=2, t=0)
            expressions.append(src)
            # Source term
            F += - src*vT*dx(source["volume"])

    # Boundary conditions
    if "boundary_conditions" in parameters["temperature"].keys():
        for bc in parameters["temperature"]["boundary_conditions"]:
            if type(bc["surfaces"]) is list:
                surfaces = bc["surfaces"]
            else:
                surfaces = [bc["surfaces"]]
            for surf in surfaces:
                if bc["type"] == "flux":
                    value = sp.printing.ccode(bc["value"])
                    value = Expression(value, degree=2, t=0)
                    # Surface flux term
                    F += - value*vT*ds(surf)
                    expressions.append(value)
                elif bc["type"] == "convective_flux":
                    h = sp.printing.ccode(bc["h_coeff"])
                    h = Expression(h, degree=2, t=0)
                    T_ext = sp.printing.ccode(bc["T_ext"])
                    T_ext = Expression(T_ext, degree=2, t=0)
                    # Surface convective flux term
                    F += h * (T - T_ext)*vT*ds(surf)
                    expressions.append(h)
                    expressions.append(T_ext)

    return F, expressions
