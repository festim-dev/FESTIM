import FESTIM
from fenics import *
import sympy as sp


def run(parameters, log_level=40):
    # Export parameters
    try:  # if parameters are in the export key
        FESTIM.export.export_parameters(parameters)
    except:
        pass

    # Check if transient
    transient = True
    if "type" in parameters["solving_parameters"].keys():
        if parameters["solving_parameters"]["type"] == "solve_transient":
            transient = True
        elif parameters["solving_parameters"]["type"] == "solve_stationary":
            transient = False
        elif "type" in parameters["solving_parameters"].keys():
            raise ValueError(
                str(parameters["solving_parameters"]["type"]) + ' unkown')

    # Declaration of variables
    dt = 0
    if transient:
        Time = parameters["solving_parameters"]["final_time"]
        initial_stepsize = parameters["solving_parameters"]["initial_stepsize"]
        dt = Constant(initial_stepsize)  # time step size
    set_log_level(log_level)

    # Mesh and refinement
    mesh = FESTIM.meshing.create_mesh(parameters["mesh_parameters"])
    # Define function space for system of concentrations and properties
    V, W = FESTIM.functionspaces_and_functions.create_function_spaces(
        mesh, len(parameters["traps"]))

    # Define and mark subdomains
    volume_markers, surface_markers = \
        FESTIM.meshing.subdomains(mesh, parameters)
    ds = Measure('ds', domain=mesh, subdomain_data=surface_markers)
    dx = Measure('dx', domain=mesh, subdomain_data=volume_markers)

    # Create functions for flux computation
    D_0, E_diff, thermal_cond, G, S = [None]*5
    if "derived_quantities" in parameters["exports"]:
        if "surface_flux" in parameters["exports"]["derived_quantities"]:
            D_0, E_diff, thermal_cond, G, S =\
                FESTIM.post_processing.create_flux_functions(
                    mesh, parameters["materials"], volume_markers)

    # Define temperature
    if parameters["temperature"]["type"] == "expression":
        T_expr = Expression(
            sp.printing.ccode(
                parameters["temperature"]['value']), t=0, degree=2)
        T = interpolate(T_expr, W)
        T_n = Function(W)
        T_n.assign(T)
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
            FESTIM.boundary_conditions.define_dirichlet_bcs_T(
                parameters, W, surface_markers)
        FT, expressions_FT = \
            FESTIM.formulations.define_variational_problem_heat_transfers(
                parameters, [T, vT, T_n], [dx, ds], dt)
        if parameters["temperature"]["type"] == "solve_stationary":
            print("Solving stationary heat equation")
            solve(FT == 0, T, bcs_T)

    # Define functions
    u, solutions = FESTIM.functionspaces_and_functions.define_functions(V)
    extrinsic_traps = \
        FESTIM.functionspaces_and_functions.define_functions_extrinsic_traps(
            W, parameters["traps"])
    testfunctions_concentrations, testfunctions_traps = \
        FESTIM.functionspaces_and_functions.define_test_functions(
            V, W, len(extrinsic_traps))

    # Initialising the solutions
    if "initial_conditions" in parameters.keys():
        initial_conditions = parameters["initial_conditions"]
    else:
        initial_conditions = []
    u_n, previous_solutions_concentrations = \
        FESTIM.initialise_solutions.initialising_solutions(
            V, initial_conditions)
    previous_solutions_traps = \
        FESTIM.initialise_solutions.initialising_extrinsic_traps(
            W, len(extrinsic_traps))

    # Boundary conditions
    print('Defining boundary conditions')
    bcs, expressions = FESTIM.boundary_conditions.apply_boundary_conditions(
        parameters["boundary_conditions"], V, surface_markers, ds,
        T)
    fluxes, expressions_fluxes = FESTIM.boundary_conditions.apply_fluxes(
        parameters["boundary_conditions"], solutions,
        testfunctions_concentrations, ds, T)

    # Define variational problem H transport
    print('Defining variational problem')
    F, expressions_F = FESTIM.formulations.formulation(
        parameters, extrinsic_traps,
        solutions, testfunctions_concentrations,
        previous_solutions_concentrations, dt, dx, T, T_n, transient=transient)
    F += fluxes

    du = TrialFunction(u.function_space())
    J = derivative(F, u, du)  # Define the Jacobian

    # Define variational problem for extrinsic traps
    if transient:
        extrinsic_formulations, expressions_form = \
            FESTIM.formulations.formulation_extrinsic_traps(
                parameters["traps"], extrinsic_traps, testfunctions_traps,
                previous_solutions_traps, dt)

    # Solution files
    files = []
    append = False
    if "xdmf" in parameters["exports"].keys():
        files = FESTIM.export.define_xdmf_files(parameters["exports"])

    derived_quantities_global = []
    if "derived_quantities" in parameters["exports"].keys():
        derived_quantities_global = \
            [FESTIM.post_processing.header_derived_quantities(parameters)]

    t = 0  # Initialising time to 0s
    timer = Timer()  # start timer

    if transient:
        #  Time-stepping
        print('Time stepping...')
        while t < Time:
            # Update current time
            t += float(dt)
            expressions = FESTIM.helpers.update_expressions(
                expressions, t)
            expressions_form = FESTIM.helpers.update_expressions(
                expressions_form, t)
            expressions_F = FESTIM.helpers.update_expressions(
                expressions_F, t)
            expressions_fluxes = FESTIM.helpers.update_expressions(
                expressions_fluxes, t)
            if parameters["temperature"]["type"] != "expression":
                expressions_FT = FESTIM.helpers.update_expressions(
                    expressions_FT, t)
                expressions_bcs_T = FESTIM.helpers.update_expressions(
                    expressions_bcs_T, t)

            else:
                T_n.assign(T)
                T_expr.t = t
                T.assign(interpolate(T_expr, W))

            # Display time
            print(str(round(t/Time*100, 2)) + ' %        ' +
                  str(round(t, 1)) + ' s' +
                  "    Ellapsed time so far: %s s" %
                  round(timer.elapsed()[0], 1),
                  end="\r")

            # Solve heat transfers
            if parameters["temperature"]["type"] == "solve_transient":
                dT = TrialFunction(T.function_space())
                JT = derivative(FT, T, dT)  # Define the Jacobian
                problem = NonlinearVariationalProblem(FT, T, bcs_T, JT)
                solver = NonlinearVariationalSolver(problem)
                solver.parameters["newton_solver"]["absolute_tolerance"] = \
                    1e-3
                solver.parameters["newton_solver"]["relative_tolerance"] = \
                    1e-10
                solver.solve()
                T_n.assign(T)

            # Solve main problem
            FESTIM.solving.solve_it(
                F, u, J, bcs, t, dt, parameters["solving_parameters"])

            # Solve extrinsic traps formulation
            for j in range(len(extrinsic_formulations)):
                solve(extrinsic_formulations[j] == 0, extrinsic_traps[j], [])

            # Post processing
            FESTIM.post_processing.run_post_processing(
                parameters,
                transient,
                u, T,
                [volume_markers, surface_markers],
                W,
                t,
                dt,
                files,
                append,
                [D_0, E_diff, thermal_cond, G, S],
                derived_quantities_global)
            append = True

            # Update previous solutions
            u_n.assign(u)
            for j in range(len(previous_solutions_traps)):
                previous_solutions_traps[j].assign(extrinsic_traps[j])
    else:
        # Solve steady state
        print('Solving steady state problem...')

        du = TrialFunction(u.function_space())
        FESTIM.solving.solve_once(
            F, u, J, bcs, parameters["solving_parameters"])

        # Post processing
        FESTIM.post_processing.run_post_processing(
            parameters,
            transient,
            u, T,
            [volume_markers, surface_markers],
            W,
            t,
            dt,
            files,
            append,
            [D_0, E_diff, thermal_cond, G, S],
            derived_quantities_global)

    # Store data in output
    output = dict()  # Final output

    # Compute error
    if u.function_space().num_sub_spaces() == 0:
        res = [u]
    else:
        res = list(u.split())
    if "error" in parameters["exports"].keys():
        error = FESTIM.post_processing.compute_error(
            parameters["exports"]["error"], t, [*res, T], mesh)
        output["error"] = error
    output["parameters"] = parameters
    output["mesh"] = mesh
    if "derived_quantities" in parameters["exports"].keys():
        output["derived_quantities"] = derived_quantities_global
        FESTIM.export.write_to_csv(parameters["exports"]["derived_quantities"],
                                   derived_quantities_global)

    # End
    print('\007s')
    return output
