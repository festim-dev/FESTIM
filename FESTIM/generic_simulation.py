import FESTIM
from fenics import *
import sympy as sp
import warnings
warnings.simplefilter('always', DeprecationWarning)


class Simulation():
    def __init__(self, parameters=None, mesh=None, materials=None, sources=[], boundary_conditions=[], traps=None, dt=None, settings=None, temperature=None, initial_conditions=[], exports=None, log_level=40):
        self.log_level = log_level

        self.settings = settings
        self.dt = dt
        if traps is None:
            self.traps = FESTIM.Traps([])
        elif type(traps) is list:
            self.traps = FESTIM.Traps(traps)
        elif isinstance(traps, FESTIM.Traps):
            self.traps = traps
        elif isinstance(traps, FESTIM.Trap):
            self.traps = FESTIM.Traps([traps])

        if type(materials) is list:
            self.materials = FESTIM.Materials(materials)
        elif isinstance(materials, FESTIM.Materials):
            self.materials = materials
        else:
            self.materials = materials

        self.boundary_conditions = boundary_conditions
        self.initial_conditions = initial_conditions
        self.T = temperature
        if exports is None:
            self.exports = FESTIM.Exports([])
        elif type(exports) is list:
            self.exports = FESTIM.Exports(exports)
        elif isinstance(exports, FESTIM.Exports):
            self.exports = exports
        self.mesh = mesh
        self.sources = sources

        # internal attributes
        self.mobile = FESTIM.Mobile()
        self.expressions = []

        self.J = None
        self.t = 0  # Initialising time to 0s
        self.timer = None

        self.dx, self.ds = None, None
        self.V, self.V_CG1, self.V_DG1 = None, None, None
        self.u = None
        self.v = None
        self.u_n = None

        self.bcs = None

        self.D = None
        self.thermal_cond = None
        self.cp = None
        self.rho = None
        self.H = None
        self.S = None

        if parameters is not None:
            msg = "The use of parameters will soon be deprecated \
                 please use the object-oriented approach instead"
            warnings.warn(msg, DeprecationWarning)
            self.create_settings(parameters)
            self.create_stepsize(parameters)
            self.create_concentration_objects(parameters)
            self.create_boundarycondition_objects(parameters)
            self.create_materials(parameters)
            self.create_temperature(parameters)
            self.create_initial_conditions(parameters)
            self.define_mesh(parameters)
            self.create_exports(parameters)
            self.create_sources_objects(parameters)

    def attribute_source_terms(self):
        field_to_object = {
            "solute": self.mobile,
            "0": self.mobile,
            0: self.mobile,
            "mobile": self.mobile,
            "T": self.T
        }
        if None not in [self.mobile, self.T]:
            for i, trap in enumerate(self.traps.traps, 1):
                field_to_object[i] = trap
                field_to_object[str(i)] = trap

            for source in self.sources:
                field_to_object[source.field].sources.append(source)

    def attribute_boundary_conditions(self):
        if self.T is not None:
            self.T.boundary_conditions = []
            for bc in self.boundary_conditions:
                if bc.component == "T":
                    self.T.boundary_conditions.append(bc)

    def create_stepsize(self, parameters):
        if self.settings.transient:
            self.dt = FESTIM.Stepsize()
            if "solving_parameters" in parameters:
                solving_parameters = parameters["solving_parameters"]
                self.dt.value.assign(solving_parameters["initial_stepsize"])
                if "adaptive_stepsize" in solving_parameters:
                    self.dt.adaptive_stepsize = {}
                    for key, val in solving_parameters["adaptive_stepsize"].items():
                        self.dt.adaptive_stepsize[key] = val
                    if "t_stop" not in solving_parameters["adaptive_stepsize"]:
                        self.dt.adaptive_stepsize["t_stop"] = None
                    if "stepsize_stop_max" not in solving_parameters["adaptive_stepsize"]:
                        self.dt.adaptive_stepsize["stepsize_stop_max"] = None

    def create_settings(self, parameters):
        my_settings = FESTIM.Settings(None, None)
        if "solving_parameters" in parameters:
            # Check if transient
            solving_parameters = parameters["solving_parameters"]
            if "type" in solving_parameters:
                if solving_parameters["type"] == "solve_transient":
                    my_settings.transient = True
                elif solving_parameters["type"] == "solve_stationary":
                    my_settings.transient = False
                    self.dt = None
                else:
                    raise ValueError(
                        str(solving_parameters["type"]) + ' unkown')

            # Declaration of variables
            if my_settings.transient:
                my_settings.final_time = solving_parameters["final_time"]

            my_settings.absolute_tolerance = solving_parameters["newton_solver"]["absolute_tolerance"]
            my_settings.relative_tolerance = solving_parameters["newton_solver"]["relative_tolerance"]
            my_settings.maximum_iterations = solving_parameters["newton_solver"]["maximum_iterations"]
            if "traps_element_type" in solving_parameters:
                my_settings.traps_element_type = solving_parameters["traps_element_type"]

            if "update_jacobian" in solving_parameters:
                my_settings.update_jacobian = solving_parameters["update_jacobian"]

            if "soret" in parameters["temperature"]:
                my_settings.soret = parameters["temperature"]["soret"]

        self.settings = my_settings

    def create_concentration_objects(self, parameters):
        self.mobile = FESTIM.Mobile()
        traps = []
        if "traps" in parameters:
            for trap in parameters["traps"]:
                if "type" in trap:
                    traps.append(FESTIM.ExtrinsicTrap(**trap))
                else:
                    traps.append(
                        FESTIM.Trap(trap["k_0"], trap["E_k"], trap["p_0"], trap["E_p"], trap["materials"], trap["density"])
                    )
        self.traps = FESTIM.Traps(traps)

    def create_sources_objects(self, parameters):
        self.sources = []
        if "traps" in parameters:
            for i, trap in enumerate(parameters["traps"], 1):
                if "source_term" in trap:
                    if type(trap["materials"]) is not list:
                        materials = [trap["materials"]]
                    else:
                        materials = trap["materials"]
                    for mat in materials:
                        self.sources.append(
                            FESTIM.Source(trap["source_term"], mat, i)
                        )
        if "source_term" in parameters:
            if isinstance(parameters["source_term"], dict):
                for mat in self.materials.materials:
                    if type(mat.id) is not list:
                        vols = [mat.id]
                    else:
                        vols = mat.id
                    for vol in vols:
                        self.sources.append(
                            FESTIM.Source(parameters["source_term"]["value"], volume=vol, field="0")
                        )
            elif isinstance(parameters["source_term"], list):
                for source_dict in parameters["source_term"]:
                    if type(source_dict["volume"]) is not list:
                        vols = [source_dict["volume"]]
                    else:
                        vols = source_dict["volume"]
                    for volume in vols:
                        self.sources.append(
                            FESTIM.Source(source_dict["value"], volume=volume, field="0")
                        )
        if "temperature" in parameters:
            if "source_term" in parameters["temperature"]:
                for source in parameters["temperature"]["source_term"]:
                    self.sources.append(
                        FESTIM.Source(source["value"], source["volume"], "T")
                    )

    def create_materials(self, parameters):
        materials = []
        if "materials" in parameters:
            for material in parameters["materials"]:
                my_mat = FESTIM.Material(**material)
                materials.append(my_mat)
        self.materials = FESTIM.Materials(materials)
        derived_quantities = {}
        if "exports" in parameters:
            if "derived_quantities" in parameters["exports"]:
                derived_quantities = parameters["exports"]["derived_quantities"]
        temp_type = "expression"  # default temperature type is expression
        if "temperature" in parameters:
            if "type" in parameters["temperature"]:
                temp_type = parameters["temperature"]["type"]
        self.materials.check_materials(temp_type, derived_quantities)

    def create_boundarycondition_objects(self, parameters):
        self.boundary_conditions = []
        if "boundary_conditions" in parameters:
            for BC in parameters["boundary_conditions"]:
                if BC["type"] in FESTIM.helpers.bc_types["dc"]:
                    my_BC = FESTIM.DirichletBC(**BC)
                elif BC["type"] not in FESTIM.helpers.bc_types["neumann"] or \
                        BC["type"] not in FESTIM.helpers.bc_types["robin"]:
                    my_BC = FESTIM.FluxBC(**BC)
                self.boundary_conditions.append(my_BC)

        if "temperature" in parameters:
            if "boundary_conditions" in parameters["temperature"]:

                BCs = parameters["temperature"]["boundary_conditions"]
                for BC in BCs:
                    if BC["type"] in FESTIM.helpers.T_bc_types["dc"]:
                        my_BC = FESTIM.DirichletBC(component="T", **BC)
                    elif BC["type"] not in FESTIM.helpers.T_bc_types["neumann"] or \
                         BC["type"] not in FESTIM.helpers.T_bc_types["robin"]:
                        my_BC = FESTIM.FluxBC(component="T", **BC)
                    self.boundary_conditions.append(my_BC)

    def create_temperature(self, parameters):
        if "temperature" in parameters:
            temp_type = parameters["temperature"]["type"]
            self.T = FESTIM.Temperature(temp_type)
            if temp_type == "expression":
                self.T.expression = parameters["temperature"]['value']
                self.T.value = parameters["temperature"]['value']
            else:
                self.T.bcs = [bc for bc in self.boundary_conditions if bc.component == "T"]
                if temp_type == "solve_transient":
                    self.T.initial_value = parameters["temperature"]["initial_condition"]
                if "source_term" in parameters["temperature"]:
                    self.T.source_term = parameters["temperature"]["source_term"]

    def create_initial_conditions(self, parameters):
        initial_conditions = []
        if "initial_conditions" in parameters.keys():
            for condition in parameters["initial_conditions"]:
                initial_conditions.append(FESTIM.InitialCondition(**condition))
        self.initial_conditions = initial_conditions

    def create_exports(self, parameters):
        self.exports = FESTIM.Exports([])
        if "exports" in parameters:
            if "xdmf" in parameters["exports"]:
                my_xdmf_exports = FESTIM.XDMFExports(**parameters["exports"]["xdmf"])
                self.exports.exports += my_xdmf_exports.xdmf_exports

            if "derived_quantities" in parameters["exports"]:
                derived_quantities = FESTIM.DerivedQuantities(**parameters["exports"]["derived_quantities"])
                self.exports.exports.append(derived_quantities)

            if "txt" in parameters["exports"]:
                txt_exports = FESTIM.TXTExports(**parameters["exports"]["txt"])
                self.exports.exports += txt_exports.exports

            if "error" in parameters["exports"]:
                for error_dict in parameters["exports"]["error"]:
                    for field, exact in zip(error_dict["fields"], error_dict["exact_solutions"]):
                        error = FESTIM.Error(field, exact, error_dict["norm"], error_dict["degree"])
                        self.exports.exports.append(error)

    def define_mesh(self, parameters):
        if "mesh_parameters" in parameters:
            mesh_parameters = parameters["mesh_parameters"]

            if "volume_file" in mesh_parameters.keys():
                self.mesh = FESTIM.MeshFromXDMF(**mesh_parameters)
            elif ("mesh" in mesh_parameters.keys() and
                    isinstance(mesh_parameters["mesh"], type(Mesh()))):
                self.mesh = FESTIM.Mesh(**mesh_parameters)
            elif "vertices" in mesh_parameters.keys():
                self.mesh = FESTIM.MeshFromVertices(mesh_parameters["vertices"])
            else:
                self.mesh = FESTIM.MeshFromRefinements(**mesh_parameters)

    def define_markers(self):
        # Define and mark subdomains
        if isinstance(self.mesh, FESTIM.Mesh):
            if isinstance(self.mesh, FESTIM.Mesh1D):
                if len(self.materials.materials) > 1:
                    self.materials.check_borders(self.mesh.size)
                self.mesh.define_markers(self.materials)

            self.volume_markers, self.surface_markers = \
                self.mesh.volume_markers, self.mesh.surface_markers
            # TODO maybe these should be attributes of self.mesh?
            self.ds = Measure(
                'ds', domain=self.mesh.mesh, subdomain_data=self.surface_markers)
            self.dx = Measure(
                'dx', domain=self.mesh.mesh, subdomain_data=self.volume_markers)

    def initialise(self):
        set_log_level(self.log_level)

        self.attribute_source_terms()
        self.attribute_boundary_conditions()

        self.define_markers()

        # Define function space for system of concentrations and properties
        self.define_function_spaces()

        # Define temperature
        self.T.create_functions(self.V_CG1, self.materials, self.dx, self.ds, self.dt)

        # Create functions for properties
        self.D, self.thermal_cond, self.cp, self.rho, self.H, self.S =\
            FESTIM.create_properties(
                self.mesh.mesh, self.materials,
                self.volume_markers, self.T.T)
        if self.S is not None:
            self.settings.chemical_pot = True

            # if the temperature is of type "solve_stationary" or "expression"
            # the solubility needs to be projected
            project_S = False
            if self.T.type == "solve_stationary":
                project_S = True
            elif self.T.type == "expression":
                if "t" not in sp.printing.ccode(self.T.value):
                    project_S = True
            if project_S:
                self.S = project(self.S, self.V_DG1)

        # Define functions
        self.initialise_concentrations()
        self.initialise_extrinsic_traps()

        # Define variational problem H transport
        self.define_variational_problem_H_transport()
        self.define_variational_problem_extrinsic_traps()

        # add measure and properties to derived_quantities
        for export in self.exports.exports:
            if isinstance(export, FESTIM.DerivedQuantities):
                export.data = [export.make_header()]
                export.assign_measures_to_quantities(self.dx, self.ds)
                export.assign_properties_to_quantities(self.D, self.S, self.thermal_cond, self.H, self.T)

    def define_function_spaces(self):
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
                self.settings.traps_element_type, mesh.ufl_cell(), order_trap)
            element = [solute] + [traps]*nb_traps
            V = FunctionSpace(mesh, MixedElement(element))
        self.V = V
        # function space for T and ext trap dens
        self.V_CG1 = FunctionSpace(mesh, 'CG', 1)
        self.V_DG1 = FunctionSpace(mesh, 'DG', 1)

        self.exports.V_DG1 = self.V_DG1

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
        field_to_component = {
            "solute": 0,
            "0": 0,
            0: 0,
        }
        for i, trap in enumerate(self.traps.traps, 1):
            field_to_component[trap.id] = i
            field_to_component[str(trap.id)] = i

        for ini in self.initial_conditions:
            value = ini.value
            component = field_to_component[ini.field]

            if self.V.num_sub_spaces() == 0:
                functionspace = self.V
            else:
                functionspace = self.V.sub(component).collapse()

            if component == 0:
                self.mobile.initialise(functionspace, value, label=ini.label, time_step=ini.time_step, S=self.S)
            else:
                trap = self.traps.get_trap(component)
                trap.initialise(functionspace, value, label=ini.label, time_step=ini.time_step)

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
                    chemical_pot=self.settings.chemical_pot,
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

        if self.settings.chemical_pot:
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
        if self.settings.transient:
            self.extrinsic_formulations, expressions_extrinsic = \
                FESTIM.formulation_extrinsic_traps(self)
            self.expressions.extend(expressions_extrinsic)

    def run(self):
        self.timer = Timer()  # start timer

        if self.settings.transient:
            # add final_time to Exports
            self.exports.final_time = self.settings.final_time

            # compute Jacobian before iterating if required
            if not self.settings.update_jacobian:
                du = TrialFunction(self.u.function_space())
                self.J = derivative(self.F, self.u, du)

            #  Time-stepping
            print('Time stepping...')
            while self.t < self.settings.final_time:
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
                self.bcs, self.settings, J=self.J)

            # Post processing
            self.run_post_processing()
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
        for export in self.exports.exports:
            if isinstance(export, FESTIM.DerivedQuantities):
                export.write()

        # End
        print('\007')
        return self.make_output()

    def iterate(self):
        # Update current time
        self.t += float(self.dt.value)
        FESTIM.update_expressions(
            self.expressions, self.t)
        FESTIM.update_expressions(
            self.T.sub_expressions, self.t)
        # TODO this could be a method of Temperature()
        if self.T.type == "expression":
            self.T.T_n.assign(self.T.T)
            self.T.expression.t = self.t
            self.T.T.assign(interpolate(self.T.expression, self.V_CG1))
        self.D._T = self.T.T
        if self.H is not None:
            self.H._T = self.T.T
        if self.thermal_cond is not None:
            self.thermal_cond._T = self.T.T
        if self.settings.chemical_pot:
            self.S._T = self.T.T

        # Display time
        simulation_percentage = round(self.t/self.settings.final_time*100, 2)
        simulation_time = round(self.t, 1)
        elapsed_time = round(self.timer.elapsed()[0], 1)
        msg = '{:.1f} %        '.format(simulation_percentage)
        msg += '{:.1e} s'.format(simulation_time)
        msg += "    Ellapsed time so far: {:.1f} s".format(elapsed_time)

        print(msg, end="\r")

        # Solve heat transfers
        # TODO this could be a method of Temperature()
        if self.T.type == "solve_transient":
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
            self.dt, self.settings, J=self.J)

        # Solve extrinsic traps formulation
        for trap in self.traps.traps:
            if isinstance(trap, FESTIM.ExtrinsicTrap):
                solve(trap.form_density == 0, trap.density[0], [])

        # Post processing
        self.run_post_processing()

        # Update previous solutions
        self.u_n.assign(self.u)
        for trap in self.traps.traps:
            if isinstance(trap, FESTIM.ExtrinsicTrap):
                trap.density_previous_solution.assign(trap.density[0])

        # avoid t > final_time
        if self.t + float(self.dt.value) > self.settings.final_time:
            self.dt.value.assign(self.settings.final_time - self.t)

    def run_post_processing(self):
        """Main post processing FESTIM function.
        """
        label_to_function = self.update_post_processing_solutions()

        self.exports.t = self.t
        self.exports.write(label_to_function, self.dt)

    def update_post_processing_solutions(self):
        if self.u.function_space().num_sub_spaces() == 0:
            res = [self.u]
        else:
            res = list(self.u.split())
        if self.settings.chemical_pot:  # c_m = theta * S
            solute = res[0]*self.S
            if self.need_projecting_solute():
                # project solute on V_DG1
                solute = project(solute, self.V_DG1)
        else:
            solute = res[0]

        self.mobile.post_processing_solution = solute

        for i, trap in enumerate(self.traps.traps, 1):
            trap.post_processing_solution = res[i]

        label_to_function = {
            "solute": self.mobile.post_processing_solution,
            "0": self.mobile.post_processing_solution,
            0: self.mobile.post_processing_solution,
            "T": self.T.T,
            "retention": sum([self.mobile.post_processing_solution] + [trap.post_processing_solution for trap in self.traps.traps])
        }
        for trap in self.traps.traps:
            label_to_function[trap.id] = trap.post_processing_solution
            label_to_function[str(trap.id)] = trap.post_processing_solution

        return label_to_function

    def need_projecting_solute(self):
        need_solute = False  # initialises to false
        for export in self.exports.exports:
            if isinstance(export, FESTIM.DerivedQuantities):
                for quantity in export.derived_quantities:
                    if isinstance(quantity, FESTIM.SurfaceFlux):
                        if quantity.field in ["0", 0, "solute"]:
                            need_solute = True
            elif isinstance(export, FESTIM.XDMFExport):
                if export.field in ["0", 0, "solute"]:
                    need_solute = True
        return need_solute

    def make_output(self):
        label_to_function = self.update_post_processing_solutions()

        for key, val in label_to_function.items():
            if not isinstance(val, Function):
                label_to_function[key] = project(val, self.V_DG1)

        output = dict()  # Final output
        # Compute error
        for export in self.exports.exports:
            if isinstance(export, FESTIM.Error):
                export.function = label_to_function[export.field]
                if "error" not in output:
                    output["error"] = []
                output["error"].append(export.compute(self.t))

        output["mesh"] = self.mesh.mesh

        # add derived quantities to output
        for export in self.exports.exports:
            if isinstance(export, FESTIM.DerivedQuantities):
                output["derived_quantities"] = export.data

        # initialise output["solutions"] with solute and temperature
        output["solutions"] = {
            "solute": label_to_function["solute"],
            "T": label_to_function["T"]
        }
        # add traps to output
        for trap in self.traps.traps:
            output["solutions"]["trap_{}".format(trap.id)] = trap.post_processing_solution
        # compute retention and add it to output
        output["solutions"]["retention"] = project(label_to_function["retention"], self.V_DG1)
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
