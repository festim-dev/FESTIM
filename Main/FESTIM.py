from fenics import *
from dolfin import *
import numpy as np
import sympy as sp
import csv
import sys
import os
import argparse


def save_as():
    '''
    - parameters : none
    - returns filedescription : string of the saving path
    '''
    valid = False
    while valid is False:
        print("Save as (.csv):")
        filedesorption = input()
        if filedesorption == '':
            filedesorption = "desorption.csv"
        if filedesorption.endswith('.csv'):
            valid = True
            try:
                with open(filedesorption, 'r') as f:
                    print('This file already exists.'
                          ' Do you want to replace it ? (y/n)')
                choice = input()
                if choice == "n" or choice == "N":
                    valid = False
                elif choice != "y" and choice != "Y":
                    valid = False
            except:
                valid = True
        else:
            print("Please enter a file ending with the extension .csv")
            valid = False
    return filedesorption


def export_TDS(filedesorption, desorption):
    '''
    - filedesorption : string, the path of the csv file.
    - desorption : list, values to be exported.
    '''
    busy = True
    while busy is True:
        try:
            with open(filedesorption, "w+") as output:
                busy = False
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows(['dTt'])
                for val in desorption:
                    writer.writerows([val])
        except:
            print("The file " + filedesorption +
                  " might currently be busy."
                  "Please close the application then press any key")
            input()
    return


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
    [_u_1, _u_2, _u_3, _u_4, _u_5, _u_6, retention] = res
    solution_dict = {
        'solute': _u_1,
        '1': _u_2,
        '2': _u_3,
        '3': _u_4,
        '4': _u_5,
        '5': _u_6,
        'retention': retention
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
                solution = solution_dict[functions[i]]
                label = labels[i]
                export_txt(
                    label + '_' + str(t) + 's', solution, W)
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
    files = list()
    for i in range(0, len(exports["xdmf"]["functions"])):
        u_file = XDMFFile(exports["xdmf"]["labels"][i] + '.xdmf')
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
    [_u_1, _u_2, _u_3, _u_4, _u_5, _u_6, retention] = res
    solution_dict = {
        'solute': _u_1,
        '1': _u_2,
        '2': _u_3,
        '3': _u_4,
        '4': _u_5,
        '5': _u_6,
        'retention': retention
    }
    for i in range(0, len(exports["xdmf"]["functions"])):
        label = exports["xdmf"]["labels"][i]
        function = exports["xdmf"]["functions"][i]
        solution = solution_dict[exports["xdmf"]["functions"][i]]
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


def create_function_spaces(mesh, element1='P', order1=1,
                           element2='P', degree2=1):
    ''' Returns FuncionSpaces for concentration and dynamic trap densities
    Arguments:
    - element1='P': string, the element of concentrations
    - order1=1: int, the order of the element of concentrations
    - element2='P': string, the element of dynamic trap densities
    - order1=2: int, the order of the element of dynamic trap densities
    '''
    element = FiniteElement(element1, interval, order1)
    mixed_element = MixedElement([element, element, element, element,
                                  element, element])
    V = FunctionSpace(mesh, mixed_element)
    W = FunctionSpace(mesh, element2, degree2)
    return V, W


def define_test_functions(V, W, number_int_traps, number_ext_traps):
    '''
    Returns the testfunctions for formulation
    Arguments:
    - V, W: FunctionSpace(), functionspaces of concentrations and
    trap densities
    - number_int_traps: int, number of intrisic traps
    - number_ext_traps: int, number of extrinsic traps
    '''
    v = TestFunction(V)
    v_1, v_2, v_3, v_4, v_5, v_6 = split(v)
    testfunctions_concentrations = [v_1, v_2, v_3, v_4, v_5, v_6]
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
    u_1, u_2, u_3, u_4, u_5, u_6 = split(u)
    solutions = [u_1, u_2, u_3, u_4, u_5, u_6]
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


def initialising_solutions(V):
    '''
    Returns the prievious solutions Function() objects for formulation
    and initialise them.
    '''
    print('Defining initial values')
    u_n, components = define_functions(V)
    expression = list()
    for u in components:
        expression.append("0")
    expression = tuple(expression)
    ini_u = Expression(expression, degree=1, t=0)
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
        ini = Expression("0", degree=1)
        previous_solutions.append(interpolate(ini, W))
    return previous_solutions


def formulation(traps, extrinsic_traps, solutions, testfunctions,
                previous_solutions, dt, dx, materials, temp, flux_, f):
    ''' Creates formulation for trapping MRE model.
    Parameters:
    - traps : dict, contains the energy, density and domains
    of the traps
    - solutions : list, contains the solution fields
    - testfunctions : list, contains the testfunctions
    - previous_solutions : list, contains the previous solution fields
    Returns:
    - F : variational formulation
    '''
    F = 0
    F += ((solutions[0] - previous_solutions[0]) / dt)*testfunctions[0]*dx
    for material in materials:
        D_0 = material['D_0']
        E_diff = material['E_diff']
        subdomain = material['id']
        F += D_0 * exp(-E_diff/k_B/temp) * \
            dot(grad(solutions[0]), grad(testfunctions[0]))*dx(subdomain)
    F += - flux_*f*testfunctions[0]*dx
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
        if type(material) is list:
            for subdomain in material:
                corresponding_material = \
                    find_material_from_id(materials, subdomain)
                D_0 = corresponding_material['D_0']
                E_diff = corresponding_material['E_diff']
                alpha = corresponding_material['alpha']
                beta = corresponding_material['beta']
                F += - D_0 * exp(-E_diff/k_B/temp)/alpha/alpha/beta * \
                    solutions[0] * (trap_density - solutions[i]) * \
                    testfunctions[i]*dx(subdomain)
                F += v_0*exp(-energy/k_B/temp)*solutions[i] * \
                    testfunctions[i]*dx(subdomain)
        else:
            subdomain = trap['materials']
            corresponding_material = \
                find_material_from_id(materials, subdomain)
            D_0 = corresponding_material['D_0']
            E_diff = corresponding_material['E_diff']
            alpha = corresponding_material['alpha']
            beta = corresponding_material['beta']
            F += - D_0 * exp(-E_diff/k_B/temp)/alpha/alpha/beta * \
                solutions[0] * (trap_density - solutions[i]) * \
                testfunctions[i]*dx(subdomain)
            F += v_0*exp(-energy/k_B/temp)*solutions[i] * \
                testfunctions[i]*dx(subdomain)
        F += ((solutions[i] - previous_solutions[i]) / dt) * \
            testfunctions[0]*dx
        i += 1
    return F


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

                F = ((solutions[i] - previous_solutions[i])/dt)*testfunctions[i]*dx
                F += -phi_0*(
                    (1 - solutions[i]/n_amax)*eta_a*f_a +
                    (1 - solutions[i]/n_bmax)*eta_b*f_b) \
                    * testfunctions[i]*dx
                formulations.append(F)
                i += 1
    return formulations, expressions


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
    measurement_dx = dx(subdomain_data=volume_markers)
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
    measurement_ds = ds(subdomain_data=surface_markers)
    return volume_markers, measurement_dx, surface_markers, measurement_ds


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


def apply_boundary_conditions(boundary_conditions, V,
                              surface_marker, ds, temp):
    '''
    Create a list of DirichletBCs.
    Arguments:
    - boundary_conditions: dict, parameters for bcs
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
    for type_BC in boundary_conditions:
        for BC in boundary_conditions[type_BC]:
            if type_BC == "dc":
                value_BC = Expression(str(BC['value']), t=0, degree=2)
            elif type_BC == "solubility":
                pressure = BC["pressure"]
                value_BC = solubility_BC(
                    pressure, BC["density"]*solubility(
                        BC["S_0"], BC["E_S"],
                        k_B, temp(0)))
                value_BC = Expression(sp.printing.ccode(value_BC), t=0,
                                      degree=2)
            expressions.append(value_BC)
            if type(BC['surface']) == list:
                for surface in BC['surface']:
                    bci = DirichletBC(V.sub(0), value_BC,
                                      surface_marker, surface)
                    bcs.append(bci)
            else:
                bci = DirichletBC(V.sub(0), value_BC,
                                  surface_marker, BC['surface'])
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


def run(parameters):
    # Declaration of variables
    implantation_time = 400
    resting_time = 50
    ramp = 8
    delta_TDS = 500
    TDS_time = int(delta_TDS / ramp) + 1

    global v_0
    v_0 = 1e13  # frequency factor s-1
    global k_B
    k_B = 8.6e-5  # Boltzmann constant
    solving_parameters = parameters["solving_parameters"]
    Time = solving_parameters["final_time"]
    num_steps = solving_parameters["num_steps"]
    dT = Time / num_steps
    dt = Constant(dT)  # time step size
    t = 0  # Initialising time to 0s
    stepsize_change_ratio = solving_parameters[
        "adaptative_time_step"][
            "stepsize_change_ratio"]
    t_stop = solving_parameters["adaptative_time_step"]["t_stop"]
    stepsize_stop_max = solving_parameters[
        "adaptative_time_step"][
            "stepsize_stop_max"]
    dt_min = solving_parameters["adaptative_time_step"]["dt_min"]
    size = parameters["mesh_parameters"]["size"]
    # Mesh and refinement
    materials = parameters["materials"]
    mesh = mesh_and_refine(parameters["mesh_parameters"])
    # Define function space for system of concentrations and properties
    V, W = create_function_spaces(mesh)

    # Define and mark subdomains
    volume_markers, dx, surface_markers, ds = subdomains(mesh, materials, size)

    # Define expressions used in variational forms
    print('Defining source terms')
    source_term = parameters["source_term"]
    flux_ = Expression(source_term["flux"], t=0, degree=2)
    f = Expression(source_term["distribution"], t=0, degree=2)
    T = parameters["temperature"]
    temp = Expression(T['value'], t=t, degree=2)
    # BCs
    print('Defining boundary conditions')
    bcs, expressions = apply_boundary_conditions(
        parameters["boundary_conditions"], V, surface_markers, ds,
        temp)

    # Define functions
    traps = parameters["traps"]

    u, solutions = define_functions(V)
    du = TrialFunction(V)
    extrinsic_traps = define_functions_extrinsic_traps(W, traps)
    testfunctions_concentrations, testfunctions_traps = \
        define_test_functions(V, W, 6, len(extrinsic_traps))
    # Initialising the solutions
    u_n, previous_solutions_concentrations = initialising_solutions(V)
    previous_solutions_traps = \
        initialising_extrinsic_traps(W, len(extrinsic_traps))

    print('Defining variational problem')
    # Define variational problem1

    F = formulation(traps, extrinsic_traps, solutions,
                    testfunctions_concentrations,
                    previous_solutions_concentrations, dt, dx, materials,
                    temp, flux_, f)
    # Define variational problem for extrinsic traps

    extrinsic_formulations, expressions_form = formulation_extrinsic_traps(
        traps, extrinsic_traps, testfunctions_traps, previous_solutions_traps,
        dt)

    # Solution files
    exports = parameters["exports"]

    filedesorption = save_as()
    # folder + exports["TDS"]["label"]+ '.csv'
    files = define_xdmf_files(exports)

    #  Time-stepping
    print('Time stepping...')
    total_n = 0
    desorption = list()
    export_total = list()
    level = 30  # 30 for WARNING 20 for INFO
    set_log_level(level)

    timer = Timer()  # start timer

    while t < Time:

        print(str(round(t/Time*100, 2)) + ' %        ' +
              str(round(t, 1)) + ' s' +
              "    Ellapsed time so far: %s s" % round(timer.elapsed()[0], 1),
              end="\r")

        J = derivative(F, u, du)  # Define the Jacobian
        problem = NonlinearVariationalProblem(F, u, bcs, J)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
        solver.parameters["newton_solver"]["absolute_tolerance"] = \
            solving_parameters['newton_solver']['absolute_tolerance']
        solver.parameters["newton_solver"]["relative_tolerance"] = \
            solving_parameters['newton_solver']['relative_tolerance']
        nb_it, converged = solver.solve()
        dt = adaptative_timestep(
            converged=converged, nb_it=nb_it, dt=dt,
            stepsize_change_ratio=stepsize_change_ratio,
            dt_min=dt_min, t=t, t_stop=t_stop,
            stepsize_stop_max=stepsize_stop_max)
        for j in range(len(extrinsic_formulations)):
            solve(extrinsic_formulations[j] == 0, extrinsic_traps[j], [])

        _u_1, _u_2, _u_3, _u_4, _u_5, _u_6 = u.split()
        res = [_u_1, _u_2, _u_3, _u_4, _u_5, _u_6]
        retention = project(_u_1)
        total_trap = 0
        for i in range(1, len(traps)+1):
            sol = res[i]
            total_trap += assemble(sol*dx)
            retention = project(retention + res[i], W)
        export_xdmf([_u_1, _u_2, _u_3, _u_4, _u_5, _u_6, retention],
                    exports, files, t)
        dt = export_profiles([_u_1, _u_2, _u_3, _u_4, _u_5, _u_6,
                             retention], exports, t, dt, W)
        total_sol = assemble(_u_1*dx)
        total = total_trap + total_sol
        desorption_rate = [-(total-total_n)/float(dt), temp(size/2), t]
        total_n = total
        if t > implantation_time+resting_time:
            desorption.append(desorption_rate)

        # Update previous solutions
        u_n.assign(u)
        for j in range(len(previous_solutions_traps)):
            previous_solutions_traps[j].assign(extrinsic_traps[j])
        # Update current time
        t += float(dt)
        temp.t += float(dt)
        flux_.t += float(dt)
        expressions = update_expressions(expressions, t)
        expressions_form = update_expressions(expressions_form, t)

    export_TDS(filedesorption, desorption)
    print('\007s')

if __name__ == "__main__":
    pass
