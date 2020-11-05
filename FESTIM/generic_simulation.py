import FESTIM
from fenics import *
import sympy as sp


class Simulation():
    def __init__(self, parameters, log_level=40):
        self.parameters = parameters
        self.log_level = log_level
        Simulation.initialise_solutions = \
            FESTIM.initialising.initialise_solutions
        Simulation.define_variational_problem_heat_transfers = \
            FESTIM.formulations.define_variational_problem_heat_transfers
        Simulation.define_dirichlet_bcs_T = \
            FESTIM.boundary_conditions.define_dirichlet_bcs_T
        Simulation.formulation = FESTIM.formulations.formulation
        Simulation.apply_boundary_conditions = \
            FESTIM.boundary_conditions.apply_boundary_conditions
        Simulation.apply_fluxes = FESTIM.boundary_conditions.apply_fluxes
        Simulation.formulation_extrinsic_traps = \
            FESTIM.formulations.formulation_extrinsic_traps
        Simulation.run_post_processing = \
            FESTIM.post_processing.run_post_processing

    def initialise(self):
        # Export parameters
        if "parameters" in self.parameters["exports"].keys():
            try:
                FESTIM.export.export_parameters(self.parameters)
            except TypeError:
                pass

        set_log_level(self.log_level)

        # Check if transient
        transient = True
        if "type" in self.parameters["solving_parameters"].keys():
            if self.parameters["solving_parameters"]["type"] == "solve_transient":
                transient = True
            elif self.parameters["solving_parameters"]["type"] == "solve_stationary":
                transient = False
            elif "type" in self.parameters["solving_parameters"].keys():
                raise ValueError(
                    str(self.parameters["solving_parameters"]["type"]) + ' unkown')
        self.transient = transient

        # Declaration of variables
        dt = 0
        if transient:
            self.final_time = self.parameters["solving_parameters"]["final_time"]
            initial_stepsize = self.parameters["solving_parameters"]["initial_stepsize"]
            dt = Constant(initial_stepsize, name="dt")  # time step size
        self.dt = dt
        # create mesh and markers
        self.define_mesh_and_markers()

        # Define function space for system of concentrations and properties
        self.define_function_spaces()

        # Define temperature
        self.define_temperature()

        # Create functions for properties
        self.D, self.thermal_cond, self.cp, self.rho, self.H, self.S =\
            FESTIM.post_processing.create_properties(
                self.mesh, self.parameters["materials"], self.volume_markers, self.T)

        # Define functions
        self.initialise_concentrations()
        self.initialise_extrinsic_traps()

        # Define variational problem H transport
        self.define_variational_problem_H_transport()
        self.define_variational_problem_extrinsic_traps()

        # Solution files
        files = []
        self.append = False
        if "xdmf" in self.parameters["exports"].keys():
            files = FESTIM.export.define_xdmf_files(self.parameters["exports"])
        self.files = files
        self.derived_quantities_global = []

    def define_mesh_and_markers(self):
        # Mesh and refinement
        self.mesh = FESTIM.meshing.create_mesh(self.parameters["mesh_parameters"])

        # Define and mark subdomains
        self.volume_markers, self.surface_markers = \
            FESTIM.meshing.subdomains(self.mesh, self.parameters)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.surface_markers)
        self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.volume_markers)

    def define_function_spaces(self):
        if "traps_element_type" in self.parameters["solving_parameters"].keys():
            trap_element = self.parameters["solving_parameters"]["traps_element_type"]
        else:
            trap_element = "CG"  # Default is CG
        self.V = FESTIM.functionspaces_and_functions.create_function_space(
            self.mesh, len(self.parameters["traps"]), element_trap=trap_element)
        self.V_CG1 = FunctionSpace(self.mesh, 'CG', 1)  # function space for T and ext trap dens
        self.V_DG1 = FunctionSpace(self.mesh, 'DG', 1)

    def define_temperature(self):
        self.T = Function(self.V_CG1, name="T")
        self.T_n = Function(self.V_CG1, name="T_n")
        self.expressions = []
        if self.parameters["temperature"]["type"] == "expression":
            self.T_expr = Expression(
                sp.printing.ccode(
                    self.parameters["temperature"]['value']), t=0, degree=2)
            self.T.assign(interpolate(self.T_expr, self.V_CG1))
            self.T_n.assign(self.T)
        else:
            # Define variational problem for heat transfers

            self.vT = TestFunction(self.V_CG1)
            if self.parameters["temperature"]["type"] == "solve_transient":
                T_ini = sp.printing.ccode(
                    self.parameters["temperature"]["initial_condition"])
                T_ini = Expression(T_ini, degree=2, t=0)
                self.T_n.assign(interpolate(T_ini, self.V_CG1))
            self.bcs_T, expressions_bcs_T = self.define_dirichlet_bcs_T()
            self.FT, expressions_FT = \
                self.define_variational_problem_heat_transfers()
            self.expressions += expressions_bcs_T + expressions_FT

            if self.parameters["temperature"]["type"] == "solve_stationary":
                print("Solving stationary heat equation")
                solve(self.FT == 0, self.T, self.bcs_T)
                self.T_n.assign(self.T)

    def initialise_concentrations(self):
        self.u = Function(self.V)

        self.v = TestFunction(self.V)

        # Initialising the solutions
        if "initial_conditions" in self.parameters.keys():
            initial_conditions = self.parameters["initial_conditions"]
        else:
            initial_conditions = []
        self.u_n = self.initialise_solutions()

    def initialise_extrinsic_traps(self):
        self.extrinsic_traps = [Function(self.V_CG1) for d in self.parameters["traps"]
                           if "type" in d.keys() if d["type"] == "extrinsic"]
        self.testfunctions_traps = [TestFunction(W) for d in self.parameters["traps"]
                               if "type" in d.keys() if d["type"] == "extrinsic"]
        self.previous_solutions_traps = \
            FESTIM.initialising.initialise_extrinsic_traps(
                self.V_CG1, len(self.extrinsic_traps))

    def define_variational_problem_H_transport(self):
        print('Defining variational problem')
        self.F, expressions_F = self.formulation()
        self.expressions += expressions_F

        # Boundary conditions
        print('Defining boundary conditions')
        self.bcs, expressions_BC = self.apply_boundary_conditions()
        fluxes, expressions_fluxes = self.apply_fluxes()
        self.F += fluxes
        self.expressions += expressions_BC + expressions_fluxes

        du = TrialFunction(self.u.function_space())
        self.J = derivative(self.F, self.u, du)  # Define the Jacobian

    def define_variational_problem_extrinsic_traps(self):
        # Define variational problem for extrinsic traps
        if self.transient:
            self.extrinsic_formulations, expressions_extrinsic = \
                self.formulation_extrinsic_traps()
            self.expressions.extend(expressions_extrinsic)

    def run(self):
        self.t = 0  # Initialising time to 0s
        timer = Timer()  # start timer

        if self.transient:
            #  Time-stepping
            print('Time stepping...')
            while self.t < self.final_time:
                # Update current time
                self.t += float(self.dt)
                FESTIM.helpers.update_expressions(
                    self.expressions, self.t)

                if self.parameters["temperature"]["type"] == "expression":
                    self.T_n.assign(self.T)
                    self.T_expr.t = self.t
                    self.T.assign(interpolate(self.T_expr, self.V_CG1))
                self.D._T = self.T
                if self.H is not None:
                    self.H._T = self.T
                if self.thermal_cond is not None:
                    self.thermal_cond._T = self.T
                if self.S is not None:
                    self.S._T = self.T

                # Display time
                print(str(round(self.t/self.final_time*100, 2)) + ' %        ' +
                      str(round(self.t, 1)) + ' s' +
                      "    Ellapsed time so far: %s s" %
                      round(timer.elapsed()[0], 1),
                      end="\r")

                # Solve heat transfers
                if self.parameters["temperature"]["type"] == "solve_transient":
                    dT = TrialFunction(self.T.function_space())
                    JT = derivative(self.FT, self.T, dT)  # Define the Jacobian
                    problem = NonlinearVariationalProblem(self.FT, self.T, self.bcs_T, JT)
                    solver = NonlinearVariationalSolver(problem)
                    solver.parameters["newton_solver"]["absolute_tolerance"] = \
                        1e-3
                    solver.parameters["newton_solver"]["relative_tolerance"] = \
                        1e-10
                    solver.solve()
                    self.T_n.assign(self.T)

                # Solve main problem
                FESTIM.solving.solve_it(
                    self.F, self.u, self.J, self.bcs, self.t,
                    self.dt, self.parameters["solving_parameters"])

                # Solve extrinsic traps formulation
                for j in range(len(self.extrinsic_formulations)):
                    solve(self.extrinsic_formulations[j] == 0, self.extrinsic_traps[j], [])

                # Post processing
                self.run_post_processing()
                self.append = True

                # Update previous solutions
                self.u_n.assign(self.u)
                for j in range(len(self.previous_solutions_traps)):
                    self.previous_solutions_traps[j].assign(self.extrinsic_traps[j])
        else:
            # Solve steady state
            print('Solving steady state problem...')

            du = TrialFunction(self.u.function_space())
            FESTIM.solving.solve_once(
                self.F, self.u, self.J, self.bcs, self.parameters["solving_parameters"])

            # Post processing
            self.run_post_processing()

        # Store data in output
        output = dict()  # Final output

        # Compute error
        if self.u.function_space().num_sub_spaces() == 0:
            res = [self.u]
        else:
            res = list(self.u.split())
        if "error" in self.parameters["exports"].keys():
            if self.S is not None:
                solute = project(res[0]*self.S, self.V_DG1)
                res[0] = solute
            error = FESTIM.post_processing.compute_error(
                self.parameters["exports"]["error"], self.t, [*res, self.T], self.mesh)
            output["error"] = error
        output["parameters"] = self.parameters
        output["mesh"] = self.mesh
        if "derived_quantities" in self.parameters["exports"].keys():
            output["derived_quantities"] = self.derived_quantities_global
            FESTIM.export.write_to_csv(self.parameters["exports"]["derived_quantities"],
                                    self.derived_quantities_global)

        # End
        print('\007s')
        return output


def run(parameters, log_level=40):
    """Main FESTIM function for complete simulations

    Arguments:
        parameters {dict} -- contains simulation parameters

    Keyword Arguments:
        log_level {int} -- set what kind of messsages are displayed
            (default: {40})
            CRITICAL  = 50, errors that may lead to data corruption
            ERROR     = 40, errors
            WARNING   = 30, warnings
            INFO      = 20, information of general interest
            PROGRESS  = 16, what's happening (broadly)
            TRACE     = 13,  what's happening (in detail)
            DBG       = 10  sundry

    Raises:
        ValueError: if solving type is unknown

    Returns:
        dict -- contains derived quantities, parameters and errors
    """
    my_sim = FESTIM.Simulation(parameters, log_level)
    my_sim.initialise()
    output = my_sim.run()
    return output
