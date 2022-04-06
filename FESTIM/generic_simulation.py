import FESTIM
from FESTIM.h_transport_problem import HTransportProblem
from fenics import *


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
        self.h_transport_problem = None
        self.t = 0  # Initialising time to 0s
        self.timer = None

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
            if bc.field == "T":
                self.T.boundary_conditions.append(bc)
            else:
                self.h_transport_problem.boundary_conditions.append(bc)

    def initialise(self):
        """Initialise the model. Defines markers, create the suitable function
        spaces, the functions, the variational forms...
        """
        set_log_level(self.log_level)

        if self.settings.chemical_pot:
            self.mobile = FESTIM.Theta()
        else:
            self.mobile = FESTIM.Mobile()
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

        # if the temperature is not time-dependent, solubility can be projected
        if self.settings.chemical_pot:
            if self.T.is_steady_state():
                self.materials.S = project(self.materials.S, self.V_DG1)
            self.mobile.S = self.materials.S
        self.h_transport_problem.initialise(self.mesh, self.materials, self.dt)

        self.exports.initialise_derived_quantities(
            self.mesh.dx, self.mesh.ds, self.materials)

    def run(self, output=False, completion_tone=False):
        """Runs the model.

        Args:
            output (bool, optional): If True, an output dict will be returned.
                Defaults to False.
            completion_tone (bool, optional): If True, a native os alert
                tone will alert user upon completion of current run. Defaults
                to False.
        Returns:
            dict: output containing solutions, mesh, derived quantities
        """
        self.timer = Timer()  # start timer

        if self.settings.transient:
            self.run_transient()
        else:
            self.run_steady()

        # End
        if completion_tone:
            print('\007')

        if output:
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
        self.h_transport_problem.update_post_processing_solutions(self.exports)

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
