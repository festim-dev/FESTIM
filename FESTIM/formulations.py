from fenics import *
import sympy as sp
import FESTIM


def formulation(parameters, extrinsic_traps, solutions, testfunctions,
                previous_solutions, dt, dx, T, T_n=None, transient=True):
    ''' Creates formulation for trapping MRE model.
    Parameters:
    - traps : dict, contains the energy, density and domains
    of the traps
    - solutions : list, contains the solution fields
    - testfunctions : list, contains the testfunctions
    - previous_solutions : list, contains the previous solution fields
    Returns:
    - F : variational formulation
    - expressions: list, contains Expression() to be updated
    '''
    k_B = FESTIM.k_B  # Boltzmann constant
    nu_0_default = 1e13  # frequency factor s-1
    expressions = []
    F = 0

    chemical_pot = False
    soret = False
    if "temperature" in parameters.keys():
        if "soret" in parameters["temperature"].keys():
            if parameters["temperature"]["soret"] is True:
                soret = True
    c_0 = solutions[0]
    c_0_n = previous_solutions[0]

    for material in parameters["materials"]:
        D_0 = material['D_0']
        E_diff = material['E_diff']
        if "S_0" in material.keys() or "E_S" in material.keys():
            chemical_pot = True
            E_S = material['E_S']
            S_0 = material['S_0']
            c_0 = solutions[0]*S_0*exp(-E_S/k_B/T)
            c_0_n = previous_solutions[0]*S_0*exp(-E_S/k_B/T_n)

        subdomain = material['id']
        if transient:
            F += ((c_0-c_0_n)/dt)*testfunctions[0]*dx(subdomain)
        F += dot(D_0 * exp(-E_diff/k_B/T)*grad(c_0),
                 grad(testfunctions[0]))*dx(subdomain)
        if soret is True:
            Q = material["H"]["free_enthalpy"]*T + material["H"]["entropy"]
            F += dot(D_0 * exp(-E_diff/k_B/T) *
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
                if isinstance(source_dict["volumes"], int):
                    volumes = [source_dict["volumes"]]
                else:
                    volumes = source_dict["volumes"]
                for vol in volumes:
                    F += - source*testfunctions[0]*dx(vol)
                expressions.append(source)
    expressions.append(T)  # Add it to the expressions to be updated
    i = 1  # index in traps
    j = 0  # index in extrinsic_traps
    for trap in parameters["traps"]:
        if 'type' in trap.keys() and trap['type'] == 'extrinsic':
            trap_density = extrinsic_traps[j]
            j += 1
        else:
            trap_density = sp.printing.ccode(trap['density'])
            trap_density = Expression(trap_density, degree=2, t=0)
            expressions.append(trap_density)

        energy = trap['energy']
        material = trap['materials']
        if transient:
            F += ((solutions[i] - previous_solutions[i]) / dt) * \
                testfunctions[i]*dx
        if type(material) is not list:
            material = [material]
        for subdomain in material:
            corresponding_material = \
                FESTIM.helpers.find_material_from_id(
                    parameters["materials"], subdomain)
            D_0 = corresponding_material['D_0']
            E_diff = corresponding_material['E_diff']
            alpha = corresponding_material['alpha']
            beta = corresponding_material['beta']
            c_0 = solutions[0]
            if chemical_pot is True:
                S_0 = corresponding_material['S_0']
                E_S = corresponding_material['E_S']
                c_0 = solutions[0]*S_0*exp(-E_S/k_B/T)
            if 'nu_0' in corresponding_material.keys():
                nu_0 = corresponding_material['nu_0']
            else:
                nu_0 = nu_0_default
            F += - D_0 * exp(-E_diff/k_B/T)/alpha/alpha/beta * \
                c_0 * (trap_density - solutions[i]) * \
                testfunctions[i]*dx(subdomain)
            F += nu_0*exp(-energy/k_B/T)*solutions[i] * \
                testfunctions[i]*dx(subdomain)
        # if a source term is set then add it to the form
        if 'source_term' in trap.keys():
            source = sp.printing.ccode(trap['source_term'])
            source = Expression(source, t=0, degree=2)
            F += -source*testfunctions[i]*dx
            expressions.append(source)

        if transient:
            F += ((solutions[i] - previous_solutions[i]) / dt) * \
                testfunctions[0]*dx
        i += 1
    return F, expressions


def formulation_extrinsic_traps(traps, solutions, testfunctions,
                                previous_solutions, dt):
    '''
    Creates a list that contains formulations to be solved during
    time stepping.
    Arguments:
    - solutions: list, contains the solutions fields
    - testfunctions: list, contains the testfunctions
    - previous_solutions: list, contains fields
    - dt: Constant(), stepsize
    - flux_, f: Expression() #todo, make this generic
    '''

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


def define_variational_problem_heat_transfers(
        parameters, functions, measurements, dt):
    '''
    Parameters:
    - parameters: dict, contains materials and temperature parameters
    - functions: list, [0]: current solution, [1]: TestFunction,
        [2]: previous solution
    - measurements: list, [0] dx, [1]: ds
    - dt: FEniCS Constant(), time step size
    Returns:
    - F: FEniCS Form(), the formulation for heat transfers problem
    - expressions: list, contains all the Expression() for later update
    '''
    print('Defining variational problem heat transfers')
    expressions = []
    dx = measurements[0]
    ds = measurements[1]
    T = functions[0]
    vT = functions[1]

    F = 0
    for mat in parameters["materials"]:
        if "thermal_cond" not in mat.keys():
            raise NameError("Missing thermal_cond key in material")
        thermal_cond = mat["thermal_cond"]
        if callable(thermal_cond):  # if thermal_cond is a function
            thermal_cond = thermal_cond(T)
        vol = mat["id"]
        if parameters["temperature"]["type"] == "solve_transient":
            T_n = functions[2]
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
    for source in parameters["temperature"]["source_term"]:
        src = sp.printing.ccode(source["value"])
        src = Expression(src, degree=2, t=0)
        expressions.append(src)
        # Source term
        F += - src*vT*dx(source["volume"])

    # Boundary conditions
    for bc in parameters["temperature"]["boundary_conditions"]:
        if type(bc["surface"]) is list:
            surfaces = bc["surface"]
        else:
            surfaces = [bc["surface"]]
        for surf in surfaces:
            if bc["type"] == "neumann":
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
