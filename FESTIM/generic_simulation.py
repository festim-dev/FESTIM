import FESTIM
from fenics import *
import sympy as sp


def run(parameters, log_level=40):
    # Export parameters
    try:  # if parameters are in the export key
        FESTIM.export.export_parameters(parameters)
    except:
        pass

    # Transient or not
    if "type" in parameters["solving_parameters"].keys():
        if parameters["solving_parameters"]["type"] == "solve_transient":
                transient = True
        elif parameters["solving_parameters"]["type"] == "solve_stationary":
            transient = False
        elif "type" in parameters["solving_parameters"].keys():
            raise ValueError(
                str(parameters["solving_parameters"]["type"]) + ' unkown')
    else:
        transient = True

    # Declaration of variables
    if transient:
        Time = parameters["solving_parameters"]["final_time"]
        initial_stepsize = parameters["solving_parameters"]["initial_stepsize"]
        dt = Constant(initial_stepsize)  # time step size
    else:
        dt = 0
    set_log_level(log_level)

    # Mesh and refinement
    size = parameters["mesh_parameters"]["size"]
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
    if "derived_quantities" in parameters["exports"]:
        if "surface_flux" in parameters["exports"]["derived_quantities"]:
            D_0, E_diff, thermal_cond =\
                FESTIM.post_processing.create_flux_functions(
                    mesh, parameters["materials"], volume_markers)

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
            FESTIM.boundary_conditions.define_dirichlet_bcs_T(
                parameters, W, surface_markers)
        FT, expressions_FT = \
            FESTIM.formulations.define_variational_problem_heat_transfers(
                parameters, [T, vT, T_n], [dx, ds], dt)
        if parameters["temperature"]["type"] == "solve_stationary":
            print("Solving stationary heat equation")
            solve(FT == 0, T, bcs_T)
            # Todo: add xdmf export for T field
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

    # Define variational problem1
    print('Defining variational problem')
    F, expressions_F = FESTIM.formulations.formulation(
        parameters, extrinsic_traps,
        solutions, testfunctions_concentrations,
        previous_solutions_concentrations, dt, dx, T, transient=transient)

    # BCs
    print('Defining boundary conditions')
    bcs, expressions = FESTIM.boundary_conditions.apply_boundary_conditions(
        parameters["boundary_conditions"], V, surface_markers, ds,
        T)
    fluxes, expressions_fluxes = FESTIM.boundary_conditions.apply_fluxes(
        parameters["boundary_conditions"], solutions,
        testfunctions_concentrations, ds, T)
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
    exports = parameters["exports"]
    if "xdmf" in parameters["exports"].keys():
        files = FESTIM.export.define_xdmf_files(exports)
        append = False

    timer = Timer()  # start timer
    error = []
    if "derived_quantities" in parameters["exports"].keys():
        derived_quantities_global = \
            [FESTIM.post_processing.header_derived_quantities(parameters)]
    if "TDS" in parameters["exports"].keys():
        inventory_n = 0
        desorption = [["t (s)", "T (K)", "d (m-2.s-1)"]]
    t = 0  # Initialising time to 0s

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
            res = list(u.split())
            retention = FESTIM.post_processing.compute_retention(u, W)
            res.append(retention)
            if isinstance(T, function.expression.Expression):
                res.append(interpolate(T, W))
            else:
                res.append(T)
            if "derived_quantities" in parameters["exports"].keys():
                derived_quantities_t = \
                    FESTIM.post_processing.derived_quantities(
                        parameters,
                        res,
                        [D_0*exp(-E_diff/T), thermal_cond],
                        [volume_markers, surface_markers])
                derived_quantities_t.insert(0, t)
                derived_quantities_global.append(derived_quantities_t)
            if "xdmf" in parameters["exports"].keys():
                FESTIM.export.export_xdmf(
                    res, exports, files, t, append=append)
                append = True
            dt = FESTIM.export.export_profiles(res, exports, t, dt, W)

            if "TDS" in parameters["exports"].keys():
                inventory = assemble(retention*dx)
                desorption_rate = \
                    [t, T(size/2), -(inventory-inventory_n)/float(dt)]
                inventory_n = inventory
                if t > parameters["exports"]["TDS"]["TDS_time"]:
                    desorption.append(desorption_rate)
            # Update previous solutions
            u_n.assign(u)
            for j in range(len(previous_solutions_traps)):
                previous_solutions_traps[j].assign(extrinsic_traps[j])
    else:
        # Solve steady state
        du = TrialFunction(u.function_space())
        FESTIM.solving.solve_once(
            F, u, J, bcs, parameters["solving_parameters"])
        res = list(u.split())
        retention = FESTIM.post_processing.compute_retention(u, W)
        res.append(retention)
        if isinstance(T, function.expression.Expression):
            res.append(interpolate(T, W))
        else:
            res.append(T)

        # Post processing
        res = list(u.split())
        retention = FESTIM.post_processing.compute_retention(u, W)
        res.append(retention)
        if "derived_quantities" in parameters["exports"].keys():
            derived_quantities_t = \
                FESTIM.post_processing.derived_quantities(
                    parameters,
                    res,
                    [D_0*exp(-E_diff/T), thermal_cond],
                    [volume_markers, surface_markers])
            derived_quantities_t.insert(0, t)
            derived_quantities_global.append(derived_quantities_t)
        if "xdmf" in parameters["exports"].keys():
            FESTIM.export.export_xdmf(
                res, exports, files, t, append=append)
            append = True
        dt = FESTIM.export.export_profiles(res, exports, t, dt, W)

    # Store data in output
    output = dict()  # Final output

    # Compute error
    if "error" in parameters["exports"].keys():
        error = FESTIM.post_processing.compute_error(
            parameters["exports"]["error"], t, u, mesh)
        output["error"] = error
    output["parameters"] = parameters
    output["mesh"] = mesh
    if "derived_quantities" in parameters["exports"].keys():
        output["derived_quantities"] = derived_quantities_global
        FESTIM.export.write_to_csv(parameters["exports"]["derived_quantities"],
                                   derived_quantities_global)

    # Export TDS
    if "TDS" in parameters["exports"].keys():
        output["TDS"] = desorption
        FESTIM.export.write_to_csv(parameters["exports"]["TDS"], desorption)

    # End
    print('\007s')
    return output
