from fenics import *
from dolfin import *
import numpy as np
import sympy as sp
import csv
import sys
import os
import argparse


def tds_to_csv(parameters, desorption):
    if "TDS" in parameters["exports"]:
        p = parameters["exports"]["TDS"]
        if "file" in p.keys():
            file_tds = ''
            if "folder" in p.keys():
                file_tds += p["folder"] + '/'
                os.makedirs(os.path.dirname(file_tds), exist_ok=True)
            file_tds += p["file"] + ".csv"
        busy = True
        while busy is True:
            try:
                with open(file_tds, "w+") as f:
                    busy = False
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerows(['dTt'])
                    for val in desorption:
                        writer.writerows([val])
            except:
                print("The file " + file_tds +
                      " might currently be busy."
                      "Please close the application then press any key")
                input()


def export_txt(filename, function, W):
    '''
    Exports a 1D function into a txt file.
    Arguments:
    - filemame : str
    - function : FEniCS Function
    - W : FunctionSpace on which the solution will be projected.
    Returns:
    - True on sucess,
    - False on failure
    '''
    export = Function(W)
    export = project(function)
    busy = True
    x = interpolate(Expression('x[0]', degree=1), W)
    while busy is True:
        try:
            np.savetxt(filename + '.txt', np.transpose(
                        [x.vector()[:], export.vector()[:]]))
            return True
        except:
            print("The file " + filename + ".txt might currently be busy."
                  "Please close the application then press any key.")
            input()
    return False


def export_profiles(res, exports, t, dt, W):
    '''
    Exports 1D profiles in txt files.
    Arguments:
    - res: list, contains FEniCS Functions
    - exports: dict, defined by define_exports()
    - t: float, time
    - dt: FEniCS Constant(), stepsize
    Returns:
    - dt: FEniCS Constant(), stepsize
    '''
    functions = exports['txt']['functions']
    labels = exports['txt']['labels']
    if len(functions) != len(labels):
        raise NameError("Number of functions to be exported "
                        "doesn't match number of labels in txt exports")
    if len(functions) > len(res):
        raise NameError("Too many functions to export "
                        "in txt exports")
    solution_dict = {
        'solute': res[0],
        'retention': res[len(res)-1]
    }
    times = sorted(exports['txt']['times'])
    end = True
    for time in times:
        if t == time:
            if times.index(time) != len(times)-1:
                next_time = times[times.index(time)+1]
                end = False
            else:
                end = True
            for i in range(len(functions)):
                try:
                    nb = int(exports["xdmf"]["functions"][i])
                    solution = res[nb]
                except:
                    solution = solution_dict[functions[i]]
                label = labels[i]
                export_txt(
                    exports["xdmf"]["folder"] + '/' + label + '_' +
                    str(t) + 's',
                    solution, W)
            break
        if t < time:
            next_time = time
            end = False
            break
    if end is False:
        if t + float(dt) > next_time:
            dt.assign(time - t)
    return dt


def define_xdmf_files(exports):
    '''
    Returns a list of XDMFFile
    Arguments:
    - exports: dict, defined by define_exports()
    '''
    if len(exports['xdmf']['functions']) != len(exports['xdmf']['labels']):
        raise NameError("Number of functions to be exported "
                        "doesn't match number of labels in xdmf exports")
    if exports["xdmf"]["folder"] == "":
        raise ValueError("folder value cannot be an empty string")
    if type(exports["xdmf"]["folder"]) is not str:
        raise TypeError("folder value must be of type str")
    files = list()
    for i in range(0, len(exports["xdmf"]["functions"])):
        u_file = XDMFFile(exports["xdmf"]["folder"]+'/' +
                          exports["xdmf"]["labels"][i] + '.xdmf')
        u_file.parameters["flush_output"] = True
        u_file.parameters["rewrite_function_mesh"] = False
        files.append(u_file)
    return files


def export_xdmf(res, exports, files, t):
    '''
    Exports the solutions fields in xdmf files.
    Arguments:
    - res: list, contains FEniCS Functions
    - exports: dict, defined by define_exports()
    - files: list, contains XDMFFile
    - t: float
    '''
    if len(exports['xdmf']['functions']) > len(res):
        raise NameError("Too many functions to export "
                        "in xdmf exports")
    solution_dict = {
        'solute': res[0],
        'retention': res[len(res)-1]
    }
    for i in range(0, len(exports["xdmf"]["functions"])):
        label = exports["xdmf"]["labels"][i]
        try:
            nb = int(exports["xdmf"]["functions"][i])
            solution = res[nb]
        except:
            try:
                solution = solution_dict[exports["xdmf"]["functions"][i]]
            except:
                raise KeyError(
                    "The key " + exports["xdmf"]["functions"][i] +
                    " is unknown.")

        solution.rename(label, "label")
        files[i].write(solution, t)
    return


def find_material_from_id(materials, id):
    ''' Returns the material from a given id
    Parameters:
    - materials : list of dicts
    - id : int
    '''
    for material in materials:
        if material['id'] == id:
            return material
            break
    print("Couldn't find ID " + str(id) + " in materials list")
    return


def create_function_spaces(mesh, nb_traps, element1='P', order1=1,
                           element2='P', degree2=1):
    ''' Returns FuncionSpaces for concentration and dynamic trap densities
    Arguments:
    - mesh: Mesh(), mesh of the functionspaces
    - nb_traps: int, number of traps
    - element1='P': string, the element of concentrations
    - order1=1: int, the order of the element of concentrations
    - element2='P': string, the element of dynamic trap densities
    - order1=2: int, the order of the element of dynamic trap densities
    '''
    if nb_traps == 0:
        V = FunctionSpace(mesh, element1, order1)
    else:
        V = VectorFunctionSpace(mesh, element1, order1, nb_traps + 1)
    W = FunctionSpace(mesh, element2, degree2)
    return V, W


def define_test_functions(V, W, number_ext_traps):
    '''
    Returns the testfunctions for formulation
    Arguments:
    - V, W: FunctionSpace(), functionspaces of concentrations and
    trap densities
    - number_int_traps: int, number of intrisic traps
    - number_ext_traps: int, number of extrinsic traps
    '''
    v = TestFunction(V)
    testfunctions_concentrations = list(split(v))
    testfunctions_extrinsic_traps = list()
    for i in range(number_ext_traps):
        testfunctions_extrinsic_traps.append(TestFunction(W))
    return testfunctions_concentrations, testfunctions_extrinsic_traps


def define_functions(V):
    '''
    Returns Function() objects for formulation
    '''
    u = Function(V)
    # Split system functions to access components
    solutions = list(split(u))
    return u, solutions


def define_functions_extrinsic_traps(W, traps):
    '''
    Returns a list of Function(W)
    Arguments:
    -W: FunctionSpace, functionspace of trap densities
    -traps: dict, contains the traps infos
    '''
    extrinsic_traps = []

    for trap in traps:
        if 'type' in trap.keys():  # Default is intrinsic
            if trap['type'] == 'extrinsic':
                extrinsic_traps.append(Function(W))  # density
    return extrinsic_traps


def initialising_solutions(V, initial_conditions):
    '''
    Returns the prievious solutions Function() objects for formulation
    and initialise them (0 by default).
    Arguments:
    - V: FunctionSpace(), function space of concentrations
    - initial_conditions: list, contains values and components
    '''
    print('Defining initial values')
    u_n, components = define_functions(V)
    # initial conditions are 0 by default
    expression = ['0'] * len(components)
    for ini in initial_conditions:
        value = ini["value"]
        value = sp.printing.ccode(value)
        expression[ini["component"]] = value
    if len(expression) == 1:
        expression = expression[0]
    else:
        expression = tuple(expression)
    ini_u = Expression(expression, degree=3, t=0)
    u_n = interpolate(ini_u, V)
    components = split(u_n)
    return u_n, components


def initialising_extrinsic_traps(W, number_of_traps):
    '''
    Returns a list of Function(W)
    Arguments:
    - W: FunctionSpace, functionspace of the extrinsic traps
    - number_of_traps: int, number of traps
    '''
    previous_solutions = []
    for i in range(number_of_traps):
        ini = Expression("0", degree=2)
        previous_solutions.append(interpolate(ini, W))
    return previous_solutions


def formulation(traps, extrinsic_traps, solutions, testfunctions,
                previous_solutions, dt, dx, materials, T, flux_):
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
    k_B = 8.6e-5  # Boltzmann constant
    v_0 = 1e13  # frequency factor s-1
    expressions = []
    F = 0
    F += ((solutions[0] - previous_solutions[0]) / dt)*testfunctions[0]*dx
    for material in materials:
        D_0 = material['D_0']
        E_diff = material['E_diff']
        subdomain = material['id']
        F += D_0 * exp(-E_diff/k_B/T) * \
            dot(grad(solutions[0]), grad(testfunctions[0]))*dx(subdomain)
    F += - flux_*testfunctions[0]*dx
    expressions.append(flux_)
    expressions.append(T)  # Add it to the expressions to be updated
    i = 1  # index in traps
    j = 0  # index in extrinsic_traps
    for trap in traps:
        if 'type' in trap.keys():
            if trap['type'] == 'extrinsic':
                trap_density = extrinsic_traps[j]
                j += 1
            else:
                trap_density = trap['density']
        else:
            trap_density = trap['density']
        energy = trap['energy']
        material = trap['materials']
        F += ((solutions[i] - previous_solutions[i]) / dt) * \
            testfunctions[i]*dx
        if type(material) is not list:
            material = [material]
        for subdomain in material:
            corresponding_material = \
                find_material_from_id(materials, subdomain)
            D_0 = corresponding_material['D_0']
            E_diff = corresponding_material['E_diff']
            alpha = corresponding_material['alpha']
            beta = corresponding_material['beta']
            F += - D_0 * exp(-E_diff/k_B/T)/alpha/alpha/beta * \
                solutions[0] * (trap_density - solutions[i]) * \
                testfunctions[i]*dx(subdomain)
            F += v_0*exp(-energy/k_B/T)*solutions[i] * \
                testfunctions[i]*dx(subdomain)
        try:  # if a source term is set then add it to the form
            source = sp.printing.ccode(trap['source_term'])
            source = Expression(source, t=0, degree=2)
            F += -source*testfunctions[i]*dx
            expressions.append(source)
        except:
            pass
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
        vol = mat["id"]
        if parameters["temperature"]["type"] == "solve_transient":
            T_n = functions[2]
            if "heat_capacity" not in mat.keys():
                raise NameError("Missing heat_capacity key in material")
            if "rho" not in mat.keys():
                raise NameError("Missing rho key in material")
            cp = mat["heat_capacity"]
            rho = mat["rho"]
            # Transien term
            F += rho*cp*(T-T_n)/dt*vT*dx(vol)
        # Diffusion term
        F += thermal_cond*dot(grad(T), grad(vT))*dx(vol)

    for source in parameters["temperature"]["source_term"]:
        src = sp.printing.ccode(source["value"])
        src = Expression(src, degree=2, t=0)
        expressions.append(src)
        # Source term
        F += - src*vT*dx(source["volume"])
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


def subdomains(mesh, materials, size):
    '''
    Iterates through the mesh and mark them
    based on their position in the domain
    Arguments:
    - mesh : the mesh
    - materials : list, contains the dictionaries of the materials
    Returns :
    - volume_markers : MeshFunction that contains the subdomains
        (0 if no domain was found)
    - measurement_dx : the measurement dx based on volume_markers
    - surface_markers : MeshFunction that contains the surfaces
    - measurement_ds : the measurement ds based on surface_markers
    '''
    volume_markers = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for cell in cells(mesh):
        for material in materials:
            if cell.midpoint().x() >= material['borders'][0] \
             and cell.midpoint().x() <= material['borders'][1]:
                volume_markers[cell] = material['id']
    surface_markers = MeshFunction(
        "size_t", mesh, mesh.topology().dim()-1, 0)
    surface_markers.set_all(0)
    i = 0
    for f in facets(mesh):
        i += 1
        x0 = f.midpoint()
        surface_markers[f] = 0
        if near(x0.x(), 0):
            surface_markers[f] = 1
        if near(x0.x(), size):
            surface_markers[f] = 2
    return volume_markers, surface_markers


def mesh_and_refine(mesh_parameters):
    '''
    Mesh and refine iteratively until meeting the refinement
    conditions.
    Arguments:
    - mesh_parameters : dict, contains initial number of cells, size,
    and refinements (number of cells and position)
    Returns:
    - mesh : the refined mesh.
    '''
    print('Meshing ...')
    initial_number_of_cells = mesh_parameters["initial_number_of_cells"]
    size = mesh_parameters["size"]
    mesh = IntervalMesh(initial_number_of_cells, 0, size)
    if "refinements" in mesh_parameters:
        for refinement in mesh_parameters["refinements"]:
            nb_cells_ref = refinement["cells"]
            refinement_point = refinement["x"]
            print("Mesh size before local refinement is " +
                  str(len(mesh.cells())))
            while len(mesh.cells()) < \
                    initial_number_of_cells + nb_cells_ref:
                cell_markers = MeshFunction(
                    "bool", mesh, mesh.topology().dim())
                cell_markers.set_all(False)
                for cell in cells(mesh):
                    if cell.midpoint().x() < refinement_point:
                        cell_markers[cell] = True
                mesh = refine(mesh, cell_markers)
            print("Mesh size after local refinement is " +
                  str(len(mesh.cells())))
            initial_number_of_cells = len(mesh.cells())
    else:
        print('No refinement parameters found')
    return mesh


def solubility(S_0, E_S, k_B, T):
    return S_0*exp(-E_S/k_B/T)


def solubility_BC(P, S):
    return P**0.5*S


def adaptative_timestep(converged, nb_it, dt, dt_min,
                        stepsize_change_ratio, t, t_stop,
                        stepsize_stop_max):
    '''
    Adapts the stepsize as function of the number of iterations of the
    solver.
    Arguments:
    - converged : bool, determines if the time step has converged.
    - nb_it : int, number of iterations
    - dt : Constant(), fenics object
    - dt_min : float, stepsize minimum value
    - stepsize_change_ration : float, stepsize change ratio
    - t : float, time
    - t_stop : float, time where adaptative time step stops
    - stepsize_stop_max : float, maximum stepsize after stop
    Returns:
    - dt : Constant(), fenics object
    '''
    while converged is False:
        dt.assign(float(dt)/stepsize_change_ratio)
        nb_it, converged = solver.solve()
        if float(dt) < dt_min:
            sys.exit('Error: stepsize reached minimal value')
    if t > t_stop:
        if float(dt) > stepsize_stop_max:
            dt.assign(stepsize_stop_max)
    else:
        if nb_it < 5:
            dt.assign(float(dt)*stepsize_change_ratio)
        else:
            dt.assign(float(dt)/stepsize_change_ratio)
    return dt


def define_dirichlet_bcs_T(parameters, V, boundaries):
    '''
    Arguments:
    - parameters: dict, contains materials and temperature parameters
    - V: FEniCS FunctionSpace(), functionspace of temperature
    - boundaries: FEniCS MeshFunction(), markers for facets.
    Returns:
    - bcs: list, contains FEniCS DirichletBC()
    - expressions: list, contains FEniCS Expression() to be updated
    '''
    bcs = []
    expressions = []
    for bc in parameters["temperature"]["boundary_conditions"]:
        if bc["type"] == "dirichlet":
            value = sp.printing.ccode(bc["value"])
            value = Expression(value, degree=2, t=0)
            expressions.append(value)
            if type(bc["surface"]) is list:
                surfaces = bc["surface"]
            else:
                surfaces = [bc["surface"]]
            for surf in surfaces:
                bci = DirichletBC(V, value, boundaries, surf)
                bcs.append(bci)
    return bcs, expressions


def apply_boundary_conditions(boundary_conditions, V,
                              surface_marker, ds, temp):
    '''
    Create a list of DirichletBCs.
    Arguments:
    - boundary_conditions: list, parameters for bcs
    - V: FunctionSpace,
    - surface_marker: MeshFunction, contains the markers for
    the different surfaces
    - ds: Measurement
    - temp: Expression, temperature.
    Returns:
    - bcs: list, contains fenics DirichletBC
    - expression: list, contains the fenics Expression
    to be updated.
    '''
    bcs = list()
    expressions = list()
    for BC in boundary_conditions:
        try:
            type_BC = BC["type"]
        except:
            raise KeyError("Missing boundary condition type key")
        if type_BC == "dc":
            value_BC = sp.printing.ccode(BC['value'])
            value_BC = Expression(value_BC, t=0, degree=4)
        elif type_BC == "solubility":
            pressure = BC["pressure"]
            value_BC = solubility_BC(
                    pressure, BC["density"]*solubility(
                        BC["S_0"], BC["E_S"],
                        k_B, T(0)))
            value_BC = Expression(sp.printing.ccode(value_BC), t=0,
                                  degree=2)
        expressions.append(value_BC)
        try:
            # Fetch the component of the BC
            component = BC["component"]
        except:
            # By default, component is solute (ie. 0)
            component = 0
        if type(BC['surface']) is not list:
            surfaces = [BC['surface']]
        else:
            surfaces = BC['surface']
        if V.num_sub_spaces() == 0:
            funspace = V
        else:  # if only one component, use subspace
            funspace = V.sub(component)
        for surface in surfaces:
                bci = DirichletBC(funspace, value_BC,
                                  surface_marker, surface)
                bcs.append(bci)

    return bcs, expressions


def update_expressions(expressions, t):
    '''
    Arguments:
    - expressions: list, contains the fenics Expression
    to be updated.
    - t: float, time.
    Update all FEniCS Expression() in expressions.
    '''
    for expression in expressions:
        expression.t = t
    return expressions


def solve_u(F, u, bcs, t, dt, solving_parameters):
    du = TrialFunction(u.function_space())
    J = derivative(F, u, du)  # Define the Jacobian
    problem = NonlinearVariationalProblem(F, u, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
    solver.parameters["newton_solver"]["absolute_tolerance"] = \
        solving_parameters['newton_solver']['absolute_tolerance']
    solver.parameters["newton_solver"]["relative_tolerance"] = \
        solving_parameters['newton_solver']['relative_tolerance']
    nb_it, converged = solver.solve()

    t_stop = solving_parameters["adaptative_time_step"]["t_stop"]
    stepsize_stop_max = \
        solving_parameters["adaptative_time_step"]["stepsize_stop_max"]
    stepsize_change_ratio = \
        solving_parameters["adaptative_time_step"]["stepsize_change_ratio"]
    dt_min = solving_parameters["adaptative_time_step"]["dt_min"]
    dt = adaptative_timestep(
        converged=converged, nb_it=nb_it, dt=dt,
        stepsize_change_ratio=stepsize_change_ratio,
        dt_min=dt_min, t=t, t_stop=t_stop,
        stepsize_stop_max=stepsize_stop_max)
    return u, dt


def compute_error(parameters, t, u_n, mesh):
    '''
    Returns a list containing the errors
    '''
    res = u_n.split()
    tab = []
    for error in parameters:
        er = []
        er.append(t)
        for i in range(len(error["exact_solution"])):
            sol = Expression(sp.printing.ccode(error["exact_solution"][i]),
                             degree=error["degree"], t=t)
            if error["norm"] == "error_max":
                vertex_values_u = res[i].compute_vertex_values(mesh)
                vertex_values_sol = sol.compute_vertex_values(mesh)
                error_max = np.max(np.abs(vertex_values_u - vertex_values_sol))
                er.append(error_max)
            else:
                error_L2 = errornorm(
                    sol, res[i], error["norm"])
                er.append(error_L2)

        tab.append(er)
    return tab


def compute_retention(u, W):
    res = list(split(u))
    if not res:  # if u is non-vector
        res = [u]
    retention = project(res[0])
    total_trap = 0
    for i in range(1, len(res)):
        sol = res[i]
        retention = project(retention + res[i], W)
    return retention


def run(parameters):
    # Declaration of variables
    Time = parameters["solving_parameters"]["final_time"]
    num_steps = parameters["solving_parameters"]["num_steps"]
    dt = Constant(Time / num_steps)  # time step size
    level = 30  # 30 for WARNING 20 for INFO
    set_log_level(level)

    # Mesh and refinement
    size = parameters["mesh_parameters"]["size"]
    mesh = mesh_and_refine(parameters["mesh_parameters"])
    # Define function space for system of concentrations and properties
    V, W = create_function_spaces(mesh, len(parameters["traps"]))

    # Define and mark subdomains
    volume_markers, surface_markers = \
        subdomains(mesh, parameters["materials"], size)
    ds = Measure('ds', domain=mesh, subdomain_data=surface_markers)
    dx = Measure('dx', domain=mesh, subdomain_data=volume_markers)

    # Define expressions used in variational forms
    print('Defining source terms')
    flux_ = Expression(
        sp.printing.ccode(parameters["source_term"]["flux"]), t=0, degree=2)

    # Define temperature
    if parameters["temperature"]["type"] == "expression":
        T = Expression(
            sp.printing.ccode(
                parameters["temperature"]['value']), t=0, degree=2)
    else:
        # Define variational problem for heat transfers
        T = Function(W, name="T")
        T_n = Function(W)
        vT = TestFunction(W)
        if parameters["temperature"]["type"] == "solve_transient":
            T_n = sp.printing.ccode(
                parameters["temperature"]["initial_condition"])
            T_n = Expression(T_n, degree=2, t=0)
            T_n = interpolate(T_n, W)
        bcs_T, expressions_bcs_T = \
            define_dirichlet_bcs_T(parameters, W, surface_markers)
        FT, expressions_FT = \
            define_variational_problem_heat_transfers(
                parameters, [T, vT, T_n], [dx, ds], dt)
        if parameters["temperature"]["type"] == "solve_stationary":
            print("Solving stationary heat equation")
            solve(FT == 0, T, bcs_T)

    # BCs
    print('Defining boundary conditions')
    bcs, expressions = apply_boundary_conditions(
        parameters["boundary_conditions"], V, surface_markers, ds,
        T)

    # Define functions

    u, solutions = define_functions(V)
    extrinsic_traps = define_functions_extrinsic_traps(W, parameters["traps"])
    testfunctions_concentrations, testfunctions_traps = \
        define_test_functions(V, W, len(extrinsic_traps))
    # Initialising the solutions
    try:
        initial_conditions = parameters["initial_conditions"]
    except:
        initial_conditions = []
    u_n, previous_solutions_concentrations = \
        initialising_solutions(V, initial_conditions)
    previous_solutions_traps = \
        initialising_extrinsic_traps(W, len(extrinsic_traps))
    print('Defining variational problem')
    # Define variational problem1

    F, expressions_F = formulation(parameters["traps"], extrinsic_traps,
                                   solutions, testfunctions_concentrations,
                                   previous_solutions_concentrations, dt, dx,
                                   parameters["materials"], T, flux_)
    # Define variational problem for extrinsic traps

    extrinsic_formulations, expressions_form = formulation_extrinsic_traps(
        parameters["traps"], extrinsic_traps, testfunctions_traps,
        previous_solutions_traps, dt)

    # Solution files
    exports = parameters["exports"]

    files = define_xdmf_files(exports)

    #  Time-stepping
    print('Time stepping...')
    inventory_n = 0
    desorption = list()
    export_total = list()

    timer = Timer()  # start timer
    error = []
    temperature = [["t (s)", "T (K)"]]
    t = 0  # Initialising time to 0s
    while t < Time:
        ## Update current time
        t += float(dt)
        expressions = update_expressions(expressions, t)
        expressions_form = update_expressions(expressions_form, t)
        expressions_F = update_expressions(expressions_F, t)
        if parameters["temperature"]["type"] != "expression":
            expressions_FT = update_expressions(expressions_FT, t)
            expressions_bcs_T = update_expressions(expressions_bcs_T, t)

        # Display time
        print(str(round(t/Time*100, 2)) + ' %        ' +
              str(round(t, 1)) + ' s' +
              "    Ellapsed time so far: %s s" % round(timer.elapsed()[0], 1),
              end="\r")

        # Solve heat transfers
        if parameters["temperature"]["type"] == "solve_transient":
            set_log_level(level)
            dT = TrialFunction(T.function_space())
            JT = derivative(FT, T, dT)  # Define the Jacobian
            problem = NonlinearVariationalProblem(FT, T, bcs_T, JT)
            solver = NonlinearVariationalSolver(problem)
            solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-3
            solver.parameters["newton_solver"]["relative_tolerance"] = 1e-10
            solver.solve()
            T_n.assign(T)
            #expressions_bcs_T = update_expressions(expressions_bcs_T, t)
            #expressions_FT = update_expressions(expressions_FT, t)
            #solve(FT == 0, T, bcs_T)
            #T_n.assign(T)
            #f.write(T, t)
        # Solve main problem
        solve_u(F, u, bcs, t, dt, parameters["solving_parameters"])
        # Solve extrinsic traps formulation
        for j in range(len(extrinsic_formulations)):
            solve(extrinsic_formulations[j] == 0, extrinsic_traps[j], [])

        # Post prossecing
        res = list(u.split())
        retention = compute_retention(u, W)
        res.append(retention)
        export_xdmf(res,
                    exports, files, t)
        dt = export_profiles(res, exports, t, dt, W)
        temperature.append([t, T(size/2)])
        inventory = assemble(retention*dx)
        desorption_rate = [-(inventory-inventory_n)/float(dt), T(size/2), t]
        inventory_n = inventory
        if t > parameters["exports"]["TDS"]["TDS_time"]:
            desorption.append(desorption_rate)
        # Update previous solutions
        u_n.assign(u)
        for j in range(len(previous_solutions_traps)):
            previous_solutions_traps[j].assign(extrinsic_traps[j])

    # Compute error
    try:
        error = compute_error(parameters["exports"]["error"], t, u_n, mesh)
    except:
        pass

    # Export TDS
    tds_to_csv(parameters, desorption)
    # Store data in output
    output = dict()  # Final output
    output["TDS"] = desorption
    output["error"] = error
    output["parameters"] = parameters
    output["mesh"] = mesh
    output["temperature"] = temperature
    # End
    print('\007s')
    return output

x, y, z, t = sp.symbols('x[0] x[1] x[2] t')

if __name__ == "__main__":
    pass
