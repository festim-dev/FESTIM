import FESTIM
from FESTIM.h_transport_problem import HTransportProblem
from fenics import *
import sympy as sp
import warnings
warnings.simplefilter('always', DeprecationWarning)


class Simulation:
    """
    Main FESTIM class representing a FESTIM model

    Attributes:
        log_level (int): set what kind of FEniCS messsages are
            displayed.
            CRITICAL  = 50, errors that may lead to data corruption
            ERROR     = 40, errors
            WARNING   = 30, warnings
            INFO      = 20, information of general interest
            PROGRESS  = 16, what's happening (broadly)
            TRACE     = 13,  what's happening (in detail)
            DBG       = 10  sundry
        settings (FESTIM.Settings): The model's settings.
        dt (FESTIM.Stepsize): The model's stepsize.
        traps (FESTIM.Traps): The model's traps.
        materials (FESTIM.Materials): The model materials.
        boundary_conditions (list of FESTIM.BoundaryCondition):
            The model's boundary conditions (temperature of H
            concentration).
        initial_conditions (list of FESTIM.InitialCondition):
            The model's initial conditions (H or T).
        T (FESTIM.Temperature): The model's temperature.
        exports (FESTIM.Exports): The model's exports
            (derived quantities, XDMF exports, txt exports...).
        mesh (FESTIM.Mesh): The mesh of the model.
        sources (list of FESTIM.Source): Volumetric sources
            (particle or heat sources).
        mobile (FESTIM.Mobile): the mobile concentration (c_m or theta)
        t (fenics.Constant): the current time of simulation
        timer (fenics.timer): the elapsed time of simulation
    """
    def __init__(
        self,
        parameters=None,
        mesh=None,
        materials=None,
        sources=[],
        boundary_conditions=[],
        traps=None,
        dt=None,
        settings=None,
        temperature=None,
        initial_conditions=[],
        exports=None,
        log_level=40
    ):
        """Inits FESTIM.Simulation

        Args:
            parameters (dict, optional): Soon to be deprecated. Defaults to
                None.
            mesh (FESTIM.Mesh, optional): The mesh of the model. Defaults to
                None.
            materials (FESTIM.Materials or [FESTIM.Material, ...], optional):
                The model materials. Defaults to None.
            sources (list of FESTIM.Source, optional): Volumetric sources
                (particle or heat sources). Defaults to [].
            boundary_conditions (list of FESTIM.BoundaryCondition, optional):
                The model's boundary conditions (temperature of H
                concentration). Defaults to None.
            traps (FESTIM.Traps or list, optional): The model's traps. Defaults
                to None.
            dt (FESTIM.Stepsize, optional): The model's stepsize. Defaults to
                None.
            settings (FESTIM.Settings, optional): The model's settings.
                Defaults to None.
            temperature (FESTIM.Temperature, optional): The model's
                temperature. Can be an expression or a heat transfer model.
                Defaults to None.
            initial_conditions (list of FESTIM.InitialCondition, optional):
                The model's initial conditions (H or T). Defaults to [].
            exports (FESTIM.Exports or list, optional): The model's exports
                (derived quantities, XDMF exports, txt exports...). Defaults
                to None.
            log_level (int, optional): set what kind of FEniCS messsages are
                displayed. Defaults to 40.
                CRITICAL  = 50, errors that may lead to data corruption
                ERROR     = 40, errors
                WARNING   = 30, warnings
                INFO      = 20, information of general interest
                PROGRESS  = 16, what's happening (broadly)
                TRACE     = 13,  what's happening (in detail)
                DBG       = 10  sundry
        """
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
        self.h_transport_problem = None
        self.t = 0  # Initialising time to 0s
        self.timer = None

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
        """Assigns the source terms (in self.sources) to the correct field
        (self.mobile, self.T, or traps)
        """
        field_to_object = {
            "solute": self.mobile,
            "0": self.mobile,
            0: self.mobile,
            "mobile": self.mobile,
            "T": self.T
        }
        for i, trap in enumerate(self.traps.traps, 1):
            field_to_object[i] = trap
            field_to_object[str(i)] = trap

        for source in self.sources:
            field_to_object[source.field].sources.append(source)

    def attribute_boundary_conditions(self):
        """Assigns boundary_conditions to mobile and T
        """
        self.T.boundary_conditions = []
        self.h_transport_problem.boundary_conditions = []

        for bc in self.boundary_conditions:
            if bc.component == "T":
                self.T.boundary_conditions.append(bc)
            else:
                self.h_transport_problem.boundary_conditions.append(bc)

    def initialise(self):
        """Initialise the model. Defines markers, create the suitable function
        spaces, the functions, the variational forms...
        """
        set_log_level(self.log_level)
        self.h_transport_problem = HTransportProblem(
            self.mobile, self.traps, self.T, self.settings,
            self.initial_conditions)
        self.attribute_source_terms()
        self.attribute_boundary_conditions()

        if isinstance(self.mesh, FESTIM.Mesh1D):
            self.mesh.define_measures(self.materials)
        else:
            self.mesh.define_measures()

        self.V_DG1 = FunctionSpace(self.mesh.mesh, 'DG', 1)
        self.exports.V_DG1 = self.V_DG1

        # Define temperature
        if isinstance(self.T, FESTIM.HeatTransferProblem):
            self.T.create_functions(self.materials, self.mesh, self.dt)
        elif isinstance(self.T, FESTIM.Temperature):
            self.T.create_functions(self.mesh)

        # Create functions for properties
        self.materials.create_properties(self.mesh.volume_markers, self.T.T)

        if self.settings.chemical_pot:
            # if the temperature is of type "solve_stationary" or "expression"
            # the solubility needs to be projected
            # TODO this should be a method of Temperature
            project_S = False
            if isinstance(self.T, FESTIM.HeatTransferProblem):
                if not self.T.transient:
                    project_S = True
            elif isinstance(self.T, FESTIM.Temperature):
                if "t" not in sp.printing.ccode(self.T.value):
                    project_S = True
            if project_S:
                self.materials.S = project(self.materials.S, self.V_DG1)

        self.h_transport_problem.initialise(self.mesh, self.materials, self.dt)

        self.exports.initialise_derived_quantities(
            self.mesh.dx, self.mesh.ds, self.materials)

    def run(self):
        """Runs the model.

        Returns:
            dict: output containing solutions, mesh, derived quantities
        """
        self.timer = Timer()  # start timer

        if self.settings.transient:
            self.run_transient()
        else:
            self.run_steady()

        # End
        print('\007')
        return self.make_output()

    def run_transient(self):
        # add final_time to Exports
        self.exports.final_time = self.settings.final_time

        # compute Jacobian before iterating if required
        if not self.settings.update_jacobian:
            self.h_transport_problem.compute_jacobian()

        #  Time-stepping
        print('Time stepping...')
        while self.t < self.settings.final_time:
            self.iterate()
        # print final message
        elapsed_time = round(self.timer.elapsed()[0], 1)
        msg = "Solved problem in {:.2f} s".format(elapsed_time)
        print(msg)

    def run_steady(self):
        # Solve steady state
        print('Solving steady state problem...')

        nb_iterations, converged = self.h_transport_problem.solve_once()

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

    def iterate(self):
        """Advance the model by one iteration
        """
        # Update current time
        self.t += float(self.dt.value)
        FESTIM.update_expressions(
            self.h_transport_problem.expressions, self.t)
        self.T.update(self.t)
        self.materials.update_properties_temperature(self.T)

        # Display time
        # TODO this should be a method
        simulation_percentage = round(self.t/self.settings.final_time*100, 2)
        simulation_time = round(self.t, 1)
        elapsed_time = round(self.timer.elapsed()[0], 1)
        msg = '{:.1f} %        '.format(simulation_percentage)
        msg += '{:.1e} s'.format(simulation_time)
        msg += "    Ellapsed time so far: {:.1f} s".format(elapsed_time)

        print(msg, end="\r")

        # update H problem
        self.h_transport_problem.update(self.t, self.dt)

        # Post processing
        self.run_post_processing()

        # avoid t > final_time
        if self.t + float(self.dt.value) > self.settings.final_time:
            self.dt.value.assign(self.settings.final_time - self.t)

    def run_post_processing(self):
        """Create post processing functions and compute/write the exports
        """
        label_to_function = self.update_post_processing_solutions()

        self.exports.t = self.t
        self.exports.write(label_to_function, self.dt)

    def update_post_processing_solutions(self):
        """Creates the post-processing functions by splitting self.u. Projects
        the function on a suitable functionspace if needed.

        Returns:
            dict: a mapping of the field ("solute", "T", "retention") to its
            post_processsing_solution
        """
        self.h_transport_problem.update_post_processing_solutions(self.materials.S, self.exports)

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

    def make_output(self):
        """Creates a dictionary with some useful information such as derived
        quantities, solutions, etc.

        Returns:
            dict: the output
        """
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

    def create_stepsize(self, parameters):
        """Creates FESTIM.Stepsize object from a parameters dict.
        To be deprecated.

        Args:
            parameters (dict): parameters dict (<= 0.7.1)
        """
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
        """Creates FESTIM.Settings object from a parameters dict.
        To be deprecated.

        Args:
            parameters (dict): parameters dict (<= 0.7.1)
        """
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
            if "materials" in parameters:
                if "S_0" in parameters["materials"][0]:
                    my_settings.chemical_pot = True
        self.settings = my_settings

    def create_concentration_objects(self, parameters):
        """Creates FESTIM.Mobile and FESTIM.Traps objects from a parameters
        dict.
        To be deprecated.

        Args:
            parameters (dict): parameters dict (<= 0.7.1)
        """
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
        """Creates a list of FESTIM.Source objects from a parameters
        dict.
        To be deprecated.

        Args:
            parameters (dict): parameters dict (<= 0.7.1)
        """
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
        """Creates a FESTIM.Materials object from a parameters
        dict.
        To be deprecated.

        Args:
            parameters (dict): parameters dict (<= 0.7.1)
        """
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
        """Creates a list of FESTIM.BoundaryCondition objects from a
        parameters dict.
        To be deprecated.

        Args:
            parameters (dict): parameters dict (<= 0.7.1)
        """
        self.boundary_conditions = []
        if "boundary_conditions" in parameters:
            for BC in parameters["boundary_conditions"]:
                bc_type = BC["type"]
                if bc_type in FESTIM.helpers.bc_types["dc"]:
                    if bc_type == "solubility":
                        my_BC = FESTIM.SievertsBC(
                            **{key: val for key, val in BC.items()
                               if key != "type"})
                    elif bc_type == "dc_imp":
                        my_BC = FESTIM.ImplantationDirichlet(
                            **{key: val for key, val in BC.items()
                               if key != "type"})
                    else:
                        my_BC = FESTIM.DirichletBC(
                            **{key: val for key, val in BC.items()
                               if key != "type"})
                elif bc_type not in FESTIM.helpers.bc_types["neumann"] or \
                        bc_type not in FESTIM.helpers.bc_types["robin"]:
                    if bc_type == "recomb":
                        my_BC = FESTIM.RecombinationFlux(
                            **{key: val for key, val in BC.items()
                                if key != "type"}
                        )
                    else:
                        my_BC = FESTIM.FluxBC(
                            **{key: val for key, val in BC.items()
                                if key != "type"}
                        )
                self.boundary_conditions.append(my_BC)

        if "temperature" in parameters:
            if "boundary_conditions" in parameters["temperature"]:

                BCs = parameters["temperature"]["boundary_conditions"]
                for BC in BCs:
                    bc_type = BC["type"]
                    if bc_type in FESTIM.helpers.T_bc_types["dc"]:
                        my_BC = FESTIM.DirichletBC(
                            component="T",
                            **{key: val for key, val in BC.items()
                               if key != "type"})
                    elif bc_type not in FESTIM.helpers.T_bc_types["neumann"] or \
                            bc_type not in FESTIM.helpers.T_bc_types["robin"]:
                        if bc_type == "convective_flux":
                            my_BC = FESTIM.ConvectiveFlux(
                                **{key: val for key, val in BC.items()
                                    if key != "type"})
                        else:
                            my_BC = FESTIM.FluxBC(
                                component="T",
                                **{key: val for key, val in BC.items()
                                    if key != "type"})
                    self.boundary_conditions.append(my_BC)

    def create_temperature(self, parameters):
        """Creates a FESTIM.Temperature object from a
        parameters dict.
        To be deprecated.

        Args:
            parameters (dict): parameters dict (<= 0.7.1)
        """
        if "temperature" in parameters:
            temp_type = parameters["temperature"]["type"]
            if temp_type == "expression":
                self.T = FESTIM.Temperature(parameters["temperature"]['value'])
                # self.T.expression = parameters["temperature"]['value']
            else:
                self.T = FESTIM.HeatTransferProblem()
                self.T.bcs = [bc for bc in self.boundary_conditions if bc.component == "T"]
                if temp_type == "solve_transient":
                    self.T.transient = True
                    self.T.initial_value = parameters["temperature"]["initial_condition"]
                elif temp_type == "solve_stationary":
                    self.T.transient = False
                if "source_term" in parameters["temperature"]:
                    self.T.source_term = parameters["temperature"]["source_term"]

    def create_initial_conditions(self, parameters):
        """Creates a list of FESTIM.InitialCondition objects from a
        parameters dict.
        To be deprecated.

        Args:
            parameters (dict): parameters dict (<= 0.7.1)
        """
        initial_conditions = []
        if "initial_conditions" in parameters.keys():
            for condition in parameters["initial_conditions"]:
                initial_conditions.append(FESTIM.InitialCondition(**condition))
        self.initial_conditions = initial_conditions

    def create_exports(self, parameters):
        """Creates a FESTIM.Exports object from a
        parameters dict.
        To be deprecated.

        Args:
            parameters (dict): parameters dict (<= 0.7.1)
        """
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
        """Creates a FESTIM.Mesh object from a
        parameters dict.
        To be deprecated.

        Args:
            parameters (dict): parameters dict (<= 0.7.1)
        """
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
