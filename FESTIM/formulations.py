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
    if "temperature" in parameters.keys():
        if "soret" in parameters["temperature"].keys():
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
        if "S_0" in material.keys() or "E_S" in material.keys():
            chemical_pot = True
            E_S = material['E_S']
            S_0 = material['S_0']
            c_0 = solutions[0]*S_0*exp(-E_S/k_B/T)
            c_0_n = previous_solutions[0]*S_0*exp(-E_S/k_B/T_n)

        subdomains = material['id']  # list of subdomains with this material
        if type(subdomains) is not list:
            subdomains = [subdomains]  # make sure subdomains is a list

        # add to the formulation F for every subdomain
        for subdomain in subdomains:
            if simulation.transient:
                F += ((c_0-c_0_n)/dt)*testfunctions[0]*dx(subdomain)
            F += dot(D_0 * exp(-E_D/k_B/T)*grad(c_0),
                     grad(testfunctions[0]))*dx(subdomain)
            if soret is True:
                Q = material["H"]["free_enthalpy"]*T + material["H"]["entropy"]
                F += dot(D_0 * exp(-E_D/k_B/T) *
                         Q * c_0 / (FESTIM.R * T**2) * grad(T),
                         grad(testfunctions[0]))*dx(subdomain)
    # Define flux
    if "source_term" in parameters.keys():
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

    all_mat_ids = [mat["id"] for mat in simulation.parameters["materials"]]

    i = 1  # index in traps
    j = 0  # index in extrinsic_traps
    for trap in parameters["traps"]:
        if 'type' in trap.keys() and trap['type'] == 'extrinsic':
            trap_density = simulation.extrinsic_traps[j]
            j += 1
        else:
            trap_density = sp.printing.ccode(trap['density'])
            trap_density = Expression(trap_density, degree=2, t=0)
            expressions.append(trap_density)

        E_k = trap['E_k']
        k_0 = trap['k_0']
        E_p = trap['E_p']
        p_0 = trap['p_0']

        trap_mat = trap['materials']
        if type(trap_mat) is not list:
            trap_mat = [trap_mat]

        if simulation.transient:
            F += ((solutions[i] - previous_solutions[i]) / dt) * \
                testfunctions[i]*dx
        else:
            # if the sim is steady state and
            # if a trap is not defined in one subdomain
            # add c_t = 0 to the form in this subdomain
            for mat_id in all_mat_ids:
                if mat_id not in trap_mat:
                    F += solutions[i]*testfunctions[i]*dx(mat_id)

        for subdomain in trap_mat:
            corresponding_material = \
                FESTIM.helpers.find_material_from_id(
                    parameters["materials"], subdomain)
            c_0 = solutions[0]
            if chemical_pot is True:
                S_0 = corresponding_material['S_0']
                E_S = corresponding_material['E_S']
                c_0 = solutions[0]*S_0*exp(-E_S/k_B/T)
            F += - k_0 * exp(-E_k/k_B/T) * c_0 \
                * (trap_density - solutions[i]) * \
                testfunctions[i]*dx(subdomain)
            F += p_0*exp(-E_p/k_B/T)*solutions[i] * \
                testfunctions[i]*dx(subdomain)
        # if a source term is set then add it to the form
        if 'source_term' in trap.keys():
            source = sp.printing.ccode(trap['source_term'])
            source = Expression(source, t=0, degree=2)
            F += -source*testfunctions[i]*dx
            expressions.append(source)

        if simulation.transient:
            F += ((solutions[i] - previous_solutions[i]) / dt) * \
                testfunctions[0]*dx
        i += 1
    return F, expressions


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
