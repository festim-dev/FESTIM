import enum
from time import time
import FESTIM
from fenics import *
import sympy as sp
import numpy as np


class Simulation():
    def __init__(self, parameters, log_level=40):
        self.parameters = parameters
        self.log_level = log_level
        self.chemical_pot = False
        self.transient = True
        self.expressions = []
        self.files = []

        self.dt = Constant(0, name="dt")
        self.nb_iterations = 0
        self.nb_iterations_between_exports = 1
        self.nb_iterations_between_export_derived_quantities = None
        self.nb_iterations_between_compute_derived_quantities = 1
        self.export_xdmf_last_only = False
        self.J = None

        self.soret = False
        self.create_concentration_objects()
        self.create_boundarycondition_objects()
        self.create_materials()
        self.define_mesh()
        self.define_markers()

        self.derived_quantities_global = [
            FESTIM.post_processing.header_derived_quantities(self)
            ]

    def create_concentration_objects(self):
        self.mobile = FESTIM.Mobile()
        traps = []
        if "traps" in self.parameters:
            for trap in self.parameters["traps"]:
                if "type" in trap:
                    traps.append(FESTIM.ExtrinsicTrap(**trap))
                else:
                    traps.append(FESTIM.Trap(**trap))
        self.traps = FESTIM.Traps(traps)

    def create_materials(self):
        materials = []
        if "materials" in self.parameters:
            for material in self.parameters["materials"]:
                my_mat = FESTIM.Material(**material)
                materials.append(my_mat)
        self.materials = FESTIM.Materials(materials)
        derived_quantities = {}
        if "exports" in self.parameters:
            if "derived_quantities" in self.parameters["exports"]:
                derived_quantities = self.parameters["exports"]["derived_quantities"]
        temp_type = "expression"  # default temperature type is expression
        if "temperature" in self.parameters:
            if "type" in self.parameters["temperature"]:
                temp_type = self.parameters["temperature"]["type"]
        self.materials.check_materials(temp_type, derived_quantities)

    def create_boundarycondition_objects(self):
        self.boundary_conditions = []
        if "boundary_conditions" in self.parameters:
            for BC in self.parameters["boundary_conditions"]:
                if BC["type"] in FESTIM.helpers.bc_types["dc"]:
                    my_BC = FESTIM.DirichletBC(**BC)
                elif BC["type"] not in FESTIM.helpers.bc_types["neumann"] or \
                        BC["type"] not in FESTIM.helpers.bc_types["robin"]:
                    my_BC = FESTIM.FluxBC(**BC)
                self.boundary_conditions.append(my_BC)

        if "temperature" in self.parameters:
            if "boundary_conditions" in self.parameters["temperature"]:

                BCs = self.parameters["temperature"]["boundary_conditions"]
                for BC in BCs:
                    if BC["type"] in FESTIM.helpers.T_bc_types["dc"]:
                        my_BC = FESTIM.DirichletBC(component="T", **BC)
                    elif BC["type"] not in FESTIM.helpers.T_bc_types["neumann"] or \
                         BC["type"] not in FESTIM.helpers.T_bc_types["robin"]:
                        my_BC = FESTIM.FluxBC(component="T", **BC)
                    self.boundary_conditions.append(my_BC)

    def initialise(self):
        # Export parameters
        if "parameters" in self.parameters["exports"].keys():
            try:
                FESTIM.export_parameters(self.parameters)
            except TypeError:
                pass

        set_log_level(self.log_level)

        # Check if transient
        solving_parameters = self.parameters["solving_parameters"]
        if "type" in solving_parameters.keys():
            if solving_parameters["type"] == "solve_transient":
                self.transient = True
            elif solving_parameters["type"] == "solve_stationary":
                self.transient = False
                self.dt = None
            else:
                raise ValueError(
                    str(solving_parameters["type"]) + ' unkown')

        # Declaration of variables
        if self.transient:
            self.final_time = solving_parameters["final_time"]
            initial_stepsize = solving_parameters["initial_stepsize"]
            self.dt.assign(initial_stepsize)  # time step size


        # Define function space for system of concentrations and properties
        self.define_function_spaces()

        # Define temperature
        self.define_temperature()

        # check if the soret effect has to be taken into account
        if "soret" in self.parameters["temperature"]:
            if self.parameters["temperature"]["soret"]:
                self.soret = True

        # Create functions for properties
        self.D, self.thermal_cond, self.cp, self.rho, self.H, self.S =\
            FESTIM.create_properties(
                self.mesh.mesh, self.materials,
                self.volume_markers, self.T.T)
        if self.S is not None:
            self.chemical_pot = True

            # if the temperature is of type "solve_stationary" or "expression"
            # the solubility needs to be projected
            project_S = False
            temp_type = self.parameters["temperature"]["type"]
            if temp_type == "solve_stationary":
                project_S = True
            elif temp_type == "expression":
                if "t" not in sp.printing.ccode(
                        self.parameters["temperature"]["value"]):
                    project_S = True
            if project_S:
                self.S = project(self.S, self.V_DG1)

        # Define functions
        self.initialise_concentrations()
        self.initialise_extrinsic_traps()

        # Define variational problem H transport
        self.define_variational_problem_H_transport()
        self.define_variational_problem_extrinsic_traps()

        # Solution files
        self.append = False
        self.exports = FESTIM.Exports([])
        if "xdmf" in self.parameters["exports"]:
            my_xdmf_exports = FESTIM.XDMFExports(**self.parameters["exports"]["xdmf"])
            self.exports.exports += my_xdmf_exports.xdmf_exports

        # TODO replace this by a derived_quantities object
        if "derived_quantities" in self.parameters["exports"]:
            derived_quant = self.parameters["exports"]["derived_quantities"]
            if "nb_iterations_between_exports" in derived_quant:
                self.nb_iterations_between_export_derived_quantities = \
                   derived_quant["nb_iterations_between_exports"]
            if "nb_iterations_between_compute" in derived_quant:
                self.nb_iterations_between_compute_derived_quantities = \
                   derived_quant["nb_iterations_between_compute"]

    def define_mesh(self):
        if "mesh_parameters" in self.parameters:
            mesh_parameters = self.parameters["mesh_parameters"]

            if "volume_file" in mesh_parameters.keys():
                self.mesh = FESTIM.MeshFromXDMF(**mesh_parameters)
            elif ("mesh" in mesh_parameters.keys() and
                    isinstance(mesh_parameters["mesh"], type(Mesh()))):
                self.mesh = FESTIM.Mesh(**mesh_parameters)
            elif "vertices" in mesh_parameters.keys():
                self.mesh = FESTIM.MeshFromVertices(mesh_parameters["vertices"])
            else:
                self.mesh = FESTIM.MeshFromRefinements(**mesh_parameters)
        else:
            self.mesh = None

    def define_markers(self):
        # Define and mark subdomains
        if isinstance(self.mesh, FESTIM.Mesh):
            if isinstance(self.mesh, FESTIM.Mesh1D):
                if len(self.materials.materials) > 1:
                    self.materials.check_borders(self.mesh.size)
                self.mesh.define_markers(self.materials)

            self.volume_markers, self.surface_markers = \
                self.mesh.volume_markers, self.mesh.surface_markers

            self.ds = Measure(
                'ds', domain=self.mesh.mesh, subdomain_data=self.surface_markers)
            self.dx = Measure(
                'dx', domain=self.mesh.mesh, subdomain_data=self.volume_markers)

    def define_function_spaces(self):
        solving_parameters = self.parameters["solving_parameters"]
        if "traps_element_type" in solving_parameters.keys():
            trap_element = solving_parameters["traps_element_type"]
        else:
            trap_element = "CG"  # Default is CG
        order_trap = 1
        element_solute, order_solute = "CG", 1

        # function space for H concentrations
        nb_traps = len(self.traps.traps)
        mesh = self.mesh.mesh
        if nb_traps == 0:
            V = FunctionSpace(mesh, element_solute, order_solute)
        else:
            solute = FiniteElement(
                element_solute, mesh.ufl_cell(), order_solute)
            traps = FiniteElement(
                trap_element, mesh.ufl_cell(), order_trap)
            element = [solute] + [traps]*nb_traps
            V = FunctionSpace(mesh, MixedElement(element))
        self.V = V
        # function space for T and ext trap dens
        self.V_CG1 = FunctionSpace(mesh, 'CG', 1)
        self.V_DG1 = FunctionSpace(mesh, 'DG', 1)

    def define_temperature(self):
        temp_type = self.parameters["temperature"]["type"]
        self.T = FESTIM.Temperature(temp_type)
        if temp_type == "expression":
            self.T.expression = self.parameters["temperature"]['value']
        else:
            self.T.bcs = [bc for bc in self.boundary_conditions if bc.component == "T"]
            if temp_type == "solve_transient":
                self.T.initial_value = self.parameters["temperature"]["initial_condition"]
            if "source_term" in self.parameters["temperature"]:
                self.T.source_term = self.parameters["temperature"]["source_term"]
        self.T.create_functions(self.V_CG1, self.materials, self.dx, self.ds, self.dt)

    def initialise_concentrations(self):
        self.u = Function(self.V, name="c")  # Function for concentrations
        self.v = TestFunction(self.V)  # TestFunction for concentrations
        self.u_n = Function(self.V, name="c_n")

        if self.V.num_sub_spaces() == 0:
            self.mobile.solution = self.u
            self.mobile.previous_solution = self.u_n
            self.mobile.test_function = self.v
        else:
            for i, concentration in enumerate([self.mobile, *self.traps.traps]):
                concentration.solution = list(split(self.u))[i]
                concentration.previous_solution = self.u_n.sub(i)
                concentration.test_function = list(split(self.v))[i]

        print('Defining initial values')

        parameters = self.parameters
        if "initial_conditions" in parameters.keys():
            initial_conditions = parameters["initial_conditions"]
        else:
            initial_conditions = []
        FESTIM.check_no_duplicates(initial_conditions)

        for ini in initial_conditions:
            value = ini['value']

            # if initial value from XDMF
            if type(value) is str and value.endswith(".xdmf"):
                label = ini['label']
                time_step = ini['time_step']
            else:
                label = None
                time_step = None
            # Default component is 0 (solute)
            if 'component' not in ini:
                ini["component"] = 0
            if self.V.num_sub_spaces() == 0:
                functionspace = self.V
            else:
                functionspace = self.V.sub(ini["component"]).collapse()

            if ini["component"] == 0:
                self.mobile.initialise(functionspace, value, label=label, time_step=time_step, S=self.S)
            else:
                trap = self.traps.get_trap(ini["component"])
                trap.initialise(functionspace, value, label=label, time_step=time_step)

        # this is needed to correctly create the formulation
        # TODO: write a test for this?
        if self.V.num_sub_spaces() != 0:
            for i, concentration in enumerate([self.mobile, *self.traps.traps]):
                concentration.previous_solution = list(split(self.u_n))[i]

    def initialise_extrinsic_traps(self):
        for trap in self.traps.traps:
            if isinstance(trap, FESTIM.ExtrinsicTrap):
                trap.density = [Function(self.V_CG1)]
                trap.density_test_function = TestFunction(self.V_CG1)
                trap.density_previous_solution = project(Constant(0), self.V_CG1)

    def define_variational_problem_H_transport(self):
        print('Defining variational problem')
        self.F, expressions_F = FESTIM.formulation(self)
        self.expressions += expressions_F

        # Boundary conditions
        print('Defining boundary conditions')
        self.create_dirichlet_bcs()
        self.create_H_fluxes()

    def create_dirichlet_bcs(self):
        self.bcs = []
        for bc in self.boundary_conditions:
            if bc.component != "T" and isinstance(bc, FESTIM.DirichletBC):
                bc.create_dirichletbc(
                    self.V, self.T.T, self.surface_markers,
                    chemical_pot=self.chemical_pot,
                    materials=self.materials,
                    volume_markers=self.volume_markers)
                self.bcs += bc.dirichlet_bc
                self.expressions += bc.sub_expressions
                self.expressions.append(bc.expression)

    def create_H_fluxes(self):
        """Modifies the formulation and adds fluxes based
        on parameters in boundary_conditions
        """

        expressions = []
        solutions = split(self.u)
        test_solute = split(self.v)[0]
        F = 0

        if self.chemical_pot:
            solute = solutions[0]*self.S
        else:
            solute = solutions[0]

        for bc in self.boundary_conditions:
            if bc.component != "T":
                if bc.type not in FESTIM.helpers.bc_types["dc"]:
                    bc.create_form_for_flux(self.T.T, solute)
                    # TODO : one day we will get rid of this huge expressions list
                    expressions += bc.sub_expressions

                    for surf in bc.surfaces:
                        F += -test_solute*bc.form*self.ds(surf)
        self.F += F
        self.expressions += expressions

    def define_variational_problem_extrinsic_traps(self):
        # Define variational problem for extrinsic traps
        if self.transient:
            self.extrinsic_formulations, expressions_extrinsic = \
                FESTIM.formulation_extrinsic_traps(self)
            self.expressions.extend(expressions_extrinsic)

    def run(self):
        self.t = 0  # Initialising time to 0s
        self.timer = Timer()  # start timer

        if self.transient:
            # compute Jacobian before iterating if required
            solving_params = self.parameters["solving_parameters"]
            if "update_jacobian" in solving_params:
                if not solving_params["update_jacobian"]:
                    du = TrialFunction(self.u.function_space())
                    self.J = derivative(self.F, self.u, du)

            #  Time-stepping
            print('Time stepping...')
            while self.t < self.final_time:
                self.iterate()
            # print final message
            elapsed_time = round(self.timer.elapsed()[0], 1)
            msg = "Solved problem in {:.2f} s".format(elapsed_time)
            print(msg)
        else:
            # Solve steady state
            print('Solving steady state problem...')

            nb_iterations, converged = FESTIM.solve_once(
                self.F, self.u,
                self.bcs, self.parameters["solving_parameters"], J=self.J)

            # Post processing
            self.update_self_processing_solutions()
            FESTIM.run_post_processing(self)
            elapsed_time = round(self.timer.elapsed()[0], 1)

            # print final message
            if converged:
                msg = "Solved problem in {:.2f} s".format(elapsed_time)
                print(msg)
            else:
                msg = "The solver diverged in "
                msg += "{:.0f} iteration(s) ({:.2f} s)".format(
                    nb_iterations, elapsed_time)
                raise ValueError(msg)

        # export derived quantities to CSV
        if "derived_quantities" in self.parameters["exports"].keys():
            FESTIM.write_to_csv(
                self.parameters["exports"]["derived_quantities"],
                self.derived_quantities_global)

        # End
        print('\007')
        return self.make_output()

    def iterate(self):
        # Update current time
        self.t += float(self.dt)
        FESTIM.update_expressions(
            self.expressions, self.t)
        FESTIM.update_expressions(
            self.T.sub_expressions, self.t)
        if self.parameters["temperature"]["type"] == "expression":
            self.T.T_n.assign(self.T.T)
            self.T.expression.t = self.t
            self.T.T.assign(interpolate(self.T.expression, self.V_CG1))
        self.D._T = self.T.T
        if self.H is not None:
            self.H._T = self.T.T
        if self.thermal_cond is not None:
            self.thermal_cond._T = self.T.T
        if self.chemical_pot:
            self.S._T = self.T.T

        # Display time
        simulation_percentage = round(self.t/self.final_time*100, 2)
        simulation_time = round(self.t, 1)
        elapsed_time = round(self.timer.elapsed()[0], 1)
        msg = '{:.1f} %        '.format(simulation_percentage)
        msg += '{:.1e} s'.format(simulation_time)
        msg += "    Ellapsed time so far: {:.1f} s".format(elapsed_time)

        print(msg, end="\r")

        # Solve heat transfers
        if self.parameters["temperature"]["type"] == "solve_transient":
            dT = TrialFunction(self.T.T.function_space())
            JT = derivative(self.T.F, self.T.T, dT)  # Define the Jacobian
            problem = NonlinearVariationalProblem(
                self.T.F, self.T.T, self.T.dirichlet_bcs, JT)
            solver = NonlinearVariationalSolver(problem)
            newton_solver_prm = solver.parameters["newton_solver"]
            newton_solver_prm["absolute_tolerance"] = 1e-3
            newton_solver_prm["relative_tolerance"] = 1e-10
            solver.solve()
            self.T.T_n.assign(self.T.T)

        # Solve main problem
        FESTIM.solve_it(
            self.F, self.u, self.bcs, self.t,
            self.dt, self.parameters["solving_parameters"], J=self.J)

        # Solve extrinsic traps formulation
        for trap in self.traps.traps:
            if isinstance(trap, FESTIM.ExtrinsicTrap):
                solve(trap.form_density == 0, trap.density[0], [])

        # Post processing
        self.update_self_processing_solutions()
        FESTIM.run_post_processing(self)

        # Update previous solutions
        self.u_n.assign(self.u)
        for trap in self.traps.traps:
            if isinstance(trap, FESTIM.ExtrinsicTrap):
                trap.density_previous_solution.assign(trap.density[0])
        self.nb_iterations += 1

        # avoid t > final_time
        if self.t + float(self.dt) > self.final_time:
            self.dt.assign(self.final_time - self.t)

    def update_self_processing_solutions(self):
        if self.u.function_space().num_sub_spaces() == 0:
            res = [self.u]
        else:
            res = list(self.u.split())
        if self.chemical_pot:  # c_m = theta * S
            theta = res[0]
            solute = project(theta*self.S, self.V_DG1)
            res[0] = solute
        else:
            solute = res[0]

        # TODO remove res
        self.res = res

        self.mobile.post_processing_solution = solute

        for i, trap in enumerate(self.traps.traps, 1):
            trap.post_processing_solution = res[i]

    def need_projecting_solute(self):
        need_solute = False  # initialises to false
        if "derived_quantities" in self.parameters["exports"].keys():
            derived_quantities_prm = self.parameters["exports"]["derived_quantities"]
            if "surface_flux" in derived_quantities_prm:
                if any(
                    x["field"] in ["0", "solute"]
                        for x in derived_quantities_prm["surface_flux"]
                        ):
                    need_solute = True
        if "xdmf" in self.parameters["exports"].keys():
            functions_to_exports = \
                self.parameters["exports"]["xdmf"]["functions"]
            if any(x in functions_to_exports for x in ["0", "solute"]):
                need_solute = True
        return need_solute

    def make_output(self):

        output = dict()  # Final output
        # Compute error
        if "error" in self.parameters["exports"].keys():
            error = FESTIM.compute_error(
                self.parameters["exports"]["error"], self.t,
                [*self.res, self.T.T], self.mesh.mesh)
            output["error"] = error

        output["parameters"] = self.parameters
        output["mesh"] = self.mesh.mesh

        # add derived quantities to output
        if "derived_quantities" in self.parameters["exports"].keys():
            output["derived_quantities"] = self.derived_quantities_global

        # initialise output["solutions"] with solute and temperature
        output["solutions"] = {
            "solute": self.mobile.post_processing_solution,
            "T": self.T.T
        }
        # add traps to output
        for trap in self.traps.traps:
            output["solutions"]["trap_{}".format(trap.id)] = trap.post_processing_solution
        # compute retention and add it to output
        output["solutions"]["retention"] = project(sum(self.res), self.V_DG1)
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
