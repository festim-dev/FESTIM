from context import FESTIM
from fenics import *
import sympy as sp


# Definition des BCs
def bc_top_H(t_implantation, t_rest, t_baking):
    t = FESTIM.t
    implantation = (t < t_implantation) * \
        1e23*2.5e-9/(2.9e-7*sp.exp(-0.39/FESTIM.k_B/1200))
    expression = implantation

    return expression


def bc_top_HT(t_implantation, t_rest, t_baking):
    t = FESTIM.t
    implantation = (t < t_implantation) * 1200
    rest = (t > t_implantation)*(t < t_implantation + t_rest) * 343
    baking = (t > t_implantation + t_rest)*350
    expression = implantation + rest + baking
    return expression


def bc_coolant_HT(t_implantation, t_rest, t_baking):
    t = FESTIM.t
    implantation = (t < t_implantation) * 373
    rest = (t > t_implantation)*(t < t_implantation + t_rest) * 343
    baking = (t > t_implantation + t_rest)*350
    expression = implantation + rest + baking

    return expression


def formulation(parameters, extrinsic_traps, solutions, testfunctions,
                previous_solutions, dt, dx, T, transient):
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
    v_0 = 1e13  # frequency factor s-1
    expressions = []
    F = 0

    for material in parameters["materials"]:
        D_0 = material['D_0']
        D_0 = material['D_0']
        E_diff = material['E_diff']
        E_S = material['E_S']
        subdomain = material['id']
        F += (S_0*exp(-E_S/k_B/T)*(solutions[0]-previous_solutions[0])/dt) *\
            testfunctions[0]*dx(subdomain)
        F += dot(D_0 * exp(-E_diff/k_B/T) *
                 grad(S_0 * exp(-E_S/k_B/T)*solutions[0]),
                 grad(testfunctions[0]))*dx(subdomain)

    i = 1  # index in traps
    for trap in parameters["traps"]:

        trap_density = sp.printing.ccode(trap['density'])
        trap_density = Expression(trap_density, degree=2, t=0)
        expressions.append(trap_density)

        energy = trap['energy']
        material = trap['materials']
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
            F += - D_0 * exp(-E_diff/k_B/T)/alpha/alpha/beta * \
                solutions[0] * (trap_density - solutions[i]) * \
                testfunctions[i]*dx(subdomain)
            F += v_0*exp(-energy/k_B/T)*solutions[i] * \
                testfunctions[i]*dx(subdomain)

        F += ((solutions[i] - previous_solutions[i]) / dt) * \
            testfunctions[0]*dx
        i += 1
    return F, expressions


def run(parameters, log_level=40):
    # Export parameters
    # FESTIM.export.export_parameters(parameters)

    transient = True

    # Declaration of variables
    dt = 0
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
    D_0, E_diff, thermal_cond, G, S =\
        FESTIM.post_processing.create_flux_functions(
            mesh, parameters["materials"], volume_markers)

    # Define variational problem for heat transfers
    T = Function(W, name="T")
    vT = TestFunction(W)
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

    # Define functions
    u, solutions = FESTIM.functionspaces_and_functions.define_functions(V)
    # S_W = Function(W)
    # S_W = S_0W*exp(-E_SW/FESTIM.k_B/T)
    # S_Cu = Function(W)
    # S_Cu = S_0Cu*exp(-E_SCu/FESTIM.k_B/T)
    # S_CuCrZr = Function(W)
    # S_CuCrZr = S_0CuCrZr*exp(-E_SCuCrZr/FESTIM.k_B/T)

    testfunctions_concentrations, testfunctions_traps = \
        FESTIM.functionspaces_and_functions.define_test_functions(
            V, W, 0)

    # Initialising the solutions
    initial_conditions = []
    u_n, previous_solutions_concentrations = \
        FESTIM.initialise_solutions.initialising_solutions(
            V, initial_conditions)
    previous_solutions_traps = \
        FESTIM.initialise_solutions.initialising_extrinsic_traps(
            W, 0)

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
        parameters, [],
        solutions, testfunctions_concentrations,
        previous_solutions_concentrations, dt, dx, T, transient=transient)
    F += fluxes

    du = TrialFunction(u.function_space())
    J = derivative(F, u, du)  # Define the Jacobian

    # Solution files
    files = []
    append = False
    files = FESTIM.export.define_xdmf_files(parameters["exports"])

    derived_quantities_global = \
        [FESTIM.post_processing.header_derived_quantities(parameters)]

    t = 0  # Initialising time to 0s
    timer = Timer()  # start timer

    #  Time-stepping
    print('Time stepping...')
    while t < Time:

        # Update current time
        t += float(dt)
        expressions = FESTIM.helpers.update_expressions(
            expressions, t)
        expressions_F = FESTIM.helpers.update_expressions(
            expressions_F, t)
        expressions_fluxes = FESTIM.helpers.update_expressions(
            expressions_fluxes, t)
        expressions_bcs_T = FESTIM.helpers.update_expressions(
            expressions_bcs_T, t)

        # Display time
        print(str(round(t/Time*100, 2)) + ' %        ' +
              str(round(t, 1)) + ' s' +
              "    Ellapsed time so far: %s s" %
              round(timer.elapsed()[0], 1),
              end="\r")

        # Solve heat transfers
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

    # End
    print('\007s')
    return


# Definition des paramètres
# atom_density  =  density(g/m3)*Na(/mol)/M(g/mol)
atom_density_W = 6.3222e28  # atomic density m^-3
atom_density_Cu = 8.4912e28  # atomic density m^-3
atom_density_CuCrZr = 2.6096e28  # atomic density m^-3

# Definition des id (doit etre les memes que dans le maillage xdmf)
id_W = 8
id_Cu = 7
id_CuCrZr = 6

id_top_surf = 9
id_coolant_surf = 10
id_left_surf = 11

# Definition des temps
t_implantation = 6000*400
t_rest = 6000*1800
t_baking = 30*24*3600

# Definition du fichier de stockage
folder = 'results/cas_test_ITER/3_traps'

# Dict parameters
parameters = {
    "mesh_parameters": {
        "mesh_file": "maillages/Mesh 10/mesh_domains.xdmf",
        "cells_file": "maillages/Mesh 10/mesh_domains.xdmf",
        "facets_file": "maillages/Mesh 10/mesh_boundaries.xdmf",
        },
    "materials": [
        {
            # Tungsten
            "D_0": 2.9e-7,
            "E_diff": 0.39,
            "S_0": 1,
            "E_S": 0,
            "alpha": 1.29e-10,
            "beta": 6*atom_density_W,
            "thermal_cond": 120,
            "heat_capacity": 1,
            "rho": 2.89e6,
            "id": id_W,
        },
        {
            # Cu
            "D_0": 6.6e-7,
            "E_diff": 0.387,
            "S_0": 1,
            "E_S": 0,
            "alpha": 3.61e-10*atom_density_Cu**0.5,
            "beta": 1,
            "thermal_cond": 350,
            "heat_capacity": 1,
            "rho": 3.67e6,
            "id": id_Cu,
        },
        {
            # CuCrZr
            "D_0": 3.92e-7,
            "E_diff": 0.418,
            "S_0": 1,
            "E_S": 0,
            "alpha": 3.61e-10*atom_density_CuCrZr**0.5,
            "beta": 1,
            "thermal_cond": 350,
            "heat_capacity": 1,
            "rho": 3.67e6,
            "id": id_CuCrZr,
        },
        ],
    "traps": [
        {
            "density": 5e-4*atom_density_W,
            "energy": 1,
            "materials": [id_W]
        },
        # {
        #     "density": 5e-3*atom_density_W*(FESTIM.y > 0.014499),
        #     "energy": 1.4,
        #     "materials": [id_W]
        # },
        {
            "density": 5e-5*atom_density_Cu,
            "energy": 0.5,
            "materials": [id_Cu]
        },
        {
            "density": 5e-5*atom_density_CuCrZr,
            "energy": 0.85,
            "materials": [id_CuCrZr]
        },
        ],
    "boundary_conditions": [
        {
            "type": "dc",
            "surface": id_top_surf,
            "value": bc_top_H(t_implantation, t_rest, t_baking)
        },
        {
            "type": "recomb",
            "surface": id_coolant_surf,
            "Kr_0": 2.9e-14,
            "E_Kr": 1.92,
            "order": 2,
        },
        # {
        #     "type": "dc",
        #     "surface": id_left_surf,
        #     "value": 0
        # },
        {
            "type": "recomb",
            "surface": [id_left_surf],
            "Kr_0": 2.9e-18,
            "E_Kr": 1.16,
            "order": 2,
        },
        ],
    "temperature": {
        "type": "solve_transient",
        "boundary_conditions": [
            {
                "type": "dirichlet",
                "value": bc_top_HT(t_implantation, t_rest, t_baking),
                "surface": id_top_surf
            },
            {
                "type": "dirichlet",
                "value": bc_coolant_HT(t_implantation, t_rest, t_baking),
                "surface": id_coolant_surf
            }
            ],
        "source_term": [
        ],
        "initial_condition": 273.15+200
        },
    "solving_parameters": {
        "final_time": t_implantation + t_rest + t_baking,
        "initial_stepsize": 1,
        "adaptive_stepsize": {
            "stepsize_change_ratio": 1.1,
            "t_stop": t_implantation,
            "stepsize_stop_max": t_rest/15,
            "dt_min": 1e-8,
            },
        "newton_solver": {
            "absolute_tolerance": 1e10,
            "relative_tolerance": 1e-9,
            "maximum_iterations": 10,
        }
        },
    "exports": {
        "xdmf": {
            "functions": ['T', 'solute', '1', '2', '3'],
            "labels": ['T', 'theta', '1', '2', '3'],
            "folder": folder
        },
        "derived_quantities": {
            "total_volume": [
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "solute"
                },
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "1"
                },
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "2"
                },
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "3"
                },
                # {
                #     "volumes": [id_W, id_Cu, id_CuCrZr],
                #     "field": "4"
                # },
                {
                    "volumes": [id_W, id_Cu, id_CuCrZr],
                    "field": "retention"
                },
            ],
            "file": "derived_quantities.csv",
            "folder": folder
        }
    }
}
run(parameters)
