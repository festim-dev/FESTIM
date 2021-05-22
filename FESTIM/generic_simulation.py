import FESTIM
from fenics import *
import sympy as sp
import numpy as np
import warnings


class Simulation():
    def __init__(self, parameters, log_level=40):
        self.parameters = parameters
        self.log_level = log_level
        self.chemical_pot = False
        self.transient = True
        self.expressions = []
        self.files = []
        self.derived_quantities_global = []
        self.dt = Constant(0, name="dt")
        self.nb_iterations = 0
        self.nb_iterations_between_exports = 1
        self.export_xdmf_last_only = False
        self.J = None

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
            else:
                raise ValueError(
                    str(solving_parameters["type"]) + ' unkown')

        # Declaration of variables
        if self.transient:
            self.final_time = solving_parameters["final_time"]
            initial_stepsize = solving_parameters["initial_stepsize"]
            self.dt.assign(initial_stepsize)  # time step size

        # create mesh and markers
        self.define_mesh()
        self.define_materials()
        self.define_markers()

        # Define function space for system of concentrations and properties
        self.define_function_spaces()

        # Define temperature
        self.define_temperature()

        # Create functions for properties
        self.D, self.thermal_cond, self.cp, self.rho, self.H, self.S =\
            FESTIM.create_properties(
                self.mesh, self.parameters["materials"],
                self.volume_markers, self.T)
        if self.S is not None:
            self.chemical_pot = True

        # Define functions
        self.initialise_concentrations()
        self.initialise_extrinsic_traps()

        # Define variational problem H transport
        self.define_variational_problem_H_transport()
        self.define_variational_problem_extrinsic_traps()

        # Solution files
        self.append = False
        exports = self.parameters["exports"]
        if "xdmf" in exports.keys():
            if "last_timestep_only" in exports["xdmf"].keys():
                self.export_xdmf_last_only = True
            self.files = FESTIM.define_xdmf_files(exports)
            if "nb_iterations_between_exports" in exports["xdmf"]:
                self.nb_iterations_between_exports = \
                   exports["xdmf"]["nb_iterations_between_exports"]

    def define_materials(self):
        materials = self.parameters["materials"]
        # check the keys in the materials are known
        for mat in materials:
            for key in mat.keys():
                if key not in FESTIM.parameters_helper["materials"]:
                    warnings.warn(
                        key + " key in materials is unknown",
                        UserWarning)

        # check the materials keys match
        if len(materials) > 1:
            old = set(materials[0].keys())
            for mat in materials:
                if not old == set(mat.keys()):
                    raise ValueError("Materials dicts keys are not the same")
                old = set(mat.keys())

        # warn about unused keys
        transient_properties = ["rho", "heat_capacity"]
        if self.parameters["temperature"]["type"] != "solve_transient":
            for key in transient_properties:
                if key in materials[0].keys():
                    warnings.warn(key + " key will be ignored", UserWarning)

        if "thermal_cond" in materials[0].keys():
            warn = True
            if self.parameters["temperature"]["type"] != "expression":
                warn = False
            elif "derived_quantities" in self.parameters["exports"].keys():
                derived_quantities = \
                    self.parameters["exports"]["derived_quantities"]
                if "surface_flux" in derived_quantities:
                    for surface_flux in derived_quantities["surface_flux"]:
                        if surface_flux["field"] == "T":
                            warn = False
            if warn:
                warnings.warn("thermal_cond key will be ignored", UserWarning)

        # check that ids are different
        ids = [mat["id"] for mat in materials]
        if not len(ids) == len(np.unique(ids)):
            raise ValueError("Some materials have the same id")

        self.materials = materials

    def define_mesh(self):

        mesh_parameters = self.parameters["mesh_parameters"]
        if "mesh_file" in mesh_parameters.keys():
            # Read volumetric mesh
            mesh = Mesh()
            XDMFFile(mesh_parameters["mesh_file"]).read(mesh)
        elif ("mesh" in mesh_parameters.keys() and
                isinstance(mesh_parameters["mesh"], type(Mesh()))):
            # use provided fenics mesh
            mesh = mesh_parameters["mesh"]
        elif "vertices" in mesh_parameters.keys():
            # mesh from list of vertices
            mesh = FESTIM.generate_mesh_from_vertices(
                mesh_parameters["vertices"])
        else:
            mesh = FESTIM.mesh_and_refine(mesh_parameters)
        self.mesh = mesh
        return mesh

    def define_markers(self):
        # Define and mark subdomains

        mesh_parameters = self.parameters["mesh_parameters"]
        if "cells_file" in mesh_parameters.keys():
            volume_markers, surface_markers = \
                FESTIM.read_subdomains_from_xdmf(
                    self.mesh,
                    mesh_parameters["cells_file"],
                    mesh_parameters["facets_file"])
        elif "meshfunction_cells" in mesh_parameters.keys():
            volume_markers = mesh_parameters["meshfunction_cells"]
            surface_markers = mesh_parameters["meshfunction_facets"]
        else:
            if "vertices" in mesh_parameters.keys():
                size = max(mesh_parameters["vertices"])
            else:
                size = mesh_parameters["size"]
            if len(self.parameters["materials"]) > 1:
                FESTIM.check_borders(
                    size, self.parameters["materials"])
            volume_markers, surface_markers = \
                FESTIM.subdomains_1D(
                    self.mesh, self.parameters["materials"], size)

        self.volume_markers, self.surface_markers = \
            volume_markers, surface_markers

        self.ds = Measure(
            'ds', domain=self.mesh, subdomain_data=self.surface_markers)
        self.dx = Measure(
            'dx', domain=self.mesh, subdomain_data=self.volume_markers)

    def define_function_spaces(self):
        solving_parameters = self.parameters["solving_parameters"]
        if "traps_element_type" in solving_parameters.keys():
            trap_element = solving_parameters["traps_element_type"]
        else:
            trap_element = "CG"  # Default is CG
        order_trap = 1
        element_solute, order_solute = "CG", 1

        # function space for H concentrations
        nb_traps = len(self.parameters["traps"])

        if nb_traps == 0:
            V = FunctionSpace(self.mesh, element_solute, order_solute)
        else:
            solute = FiniteElement(
                element_solute, self.mesh.ufl_cell(), order_solute)
            traps = FiniteElement(
                trap_element, self.mesh.ufl_cell(), order_trap)
            element = [solute] + [traps]*nb_traps
            V = FunctionSpace(self.mesh, MixedElement(element))
        self.V = V
        # function space for T and ext trap dens
        self.V_CG1 = FunctionSpace(self.mesh, 'CG', 1)
        self.V_DG1 = FunctionSpace(self.mesh, 'DG', 1)

    def define_temperature(self):
        self.T = Function(self.V_CG1, name="T")
        self.T_n = Function(self.V_CG1, name="T_n")

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
            self.bcs_T, expressions_bcs_T = FESTIM.define_dirichlet_bcs_T(self)
            self.FT, expressions_FT = \
                FESTIM.define_variational_problem_heat_transfers(self)
            self.expressions += expressions_bcs_T + expressions_FT

            if self.parameters["temperature"]["type"] == "solve_stationary":
                print("Solving stationary heat equation")
                solve(self.FT == 0, self.T, self.bcs_T)
                self.T_n.assign(self.T)

    def initialise_concentrations(self):
        self.u = Function(self.V)  # Function for concentrations

        self.v = TestFunction(self.V)  # TestFunction for concentrations

        if hasattr(self, "S"):
            S = self.S
        else:
            S = None

        print('Defining initial values')
        V = self.V
        u_n = Function(V)
        components = list(split(u_n))

        parameters = self.parameters
        if "initial_conditions" in parameters.keys():
            initial_conditions = parameters["initial_conditions"]
        else:
            initial_conditions = []
        FESTIM.check_no_duplicates(initial_conditions)

        for ini in initial_conditions:
            if 'component' not in ini.keys():
                ini["component"] = 0
            if type(ini['value']) == str and ini['value'].endswith(".xdmf"):
                comp = FESTIM.read_from_xdmf(ini, V)
            else:
                value = ini["value"]
                value = sp.printing.ccode(value)
                comp = Expression(value, degree=3, t=0)

            if ini["component"] == 0 and self.chemical_pot:
                comp = comp/S  # variable change
            if V.num_sub_spaces() > 0:
                if ini["component"] == 0 and self.chemical_pot:
                    # Product must be projected
                    comp = project(
                        comp, V.sub(ini["component"]).collapse())
                else:
                    comp = interpolate(
                        comp, V.sub(ini["component"]).collapse())
                assign(u_n.sub(ini["component"]), comp)
            else:
                if ini["component"] == 0 and self.chemical_pot:
                    u_n = project(comp, V)
                else:
                    u_n = interpolate(comp, V)
        self.u_n = u_n

    def initialise_extrinsic_traps(self):
        traps = self.parameters["traps"]
        self.extrinsic_traps = [Function(self.V_CG1) for d in traps
                                if "type" in d.keys() if
                                d["type"] == "extrinsic"]
        self.testfunctions_traps = [TestFunction(self.V_CG1) for d in traps
                                    if "type" in d.keys() if
                                    d["type"] == "extrinsic"]

        self.previous_solutions_traps = []
        for i in range(len(self.extrinsic_traps)):
            ini = Expression("0", degree=2)
            self.previous_solutions_traps.append(interpolate(ini, self.V_CG1))

    def define_variational_problem_H_transport(self):
        print('Defining variational problem')
        self.F, expressions_F = FESTIM.formulation(self)
        self.expressions += expressions_F

        # Boundary conditions
        print('Defining boundary conditions')
        self.bcs, expressions_BC = FESTIM.apply_boundary_conditions(self)
        fluxes, expressions_fluxes = FESTIM.apply_fluxes(self)
        self.F += fluxes
        self.expressions += expressions_BC + expressions_fluxes

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
        else:
            # Solve steady state
            print('Solving steady state problem...')

            FESTIM.solve_once(
                self.F, self.u,
                self.bcs, self.parameters["solving_parameters"], J=self.J)

            # Post processing
            FESTIM.run_post_processing(self)
            elapsed_time = round(self.timer.elapsed()[0], 1)
            print("Solved problem in {:.2f} s".format(elapsed_time))

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
        simulation_percentage = round(self.t/self.final_time*100, 2)
        simulation_time = round(self.t, 1)
        elapsed_time = round(self.timer.elapsed()[0], 1)
        msg = '{:.1f} %        '.format(simulation_percentage)
        msg += '{:.1e} s'.format(simulation_time)
        msg += "    Ellapsed time so far: {:.1f} s".format(elapsed_time)

        print(msg, end="\r")

        # Solve heat transfers
        if self.parameters["temperature"]["type"] == "solve_transient":
            dT = TrialFunction(self.T.function_space())
            JT = derivative(self.FT, self.T, dT)  # Define the Jacobian
            problem = NonlinearVariationalProblem(
                self.FT, self.T, self.bcs_T, JT)
            solver = NonlinearVariationalSolver(problem)
            newton_solver_prm = solver.parameters["newton_solver"]
            newton_solver_prm["absolute_tolerance"] = 1e-3
            newton_solver_prm["relative_tolerance"] = 1e-10
            solver.solve()
            self.T_n.assign(self.T)

        # Solve main problem
        FESTIM.solve_it(
            self.F, self.u, self.bcs, self.t,
            self.dt, self.parameters["solving_parameters"], J=self.J)

        # Solve extrinsic traps formulation
        for j, form in enumerate(self.extrinsic_formulations):
            solve(form == 0, self.extrinsic_traps[j], [])

        # Post processing
        FESTIM.run_post_processing(self)
        self.append = True

        # Update previous solutions
        self.u_n.assign(self.u)
        for j, prev_sol in enumerate(self.previous_solutions_traps):
            prev_sol.assign(self.extrinsic_traps[j])
        self.nb_iterations += 1

        # avoid t > final_time
        if self.t + float(self.dt) > self.final_time:
            self.dt.assign(self.final_time - self.t)

    def make_output(self):

        if self.u.function_space().num_sub_spaces() == 0:
            res = [self.u]
        else:
            res = list(self.u.split())

        if self.chemical_pot:  # c_m = theta * S
            solute = project(res[0]*self.S, self.V_DG1)
            res[0] = solute

        output = dict()  # Final output
        # Compute error
        if "error" in self.parameters["exports"].keys():
            error = FESTIM.compute_error(
                self.parameters["exports"]["error"], self.t,
                [*res, self.T], self.mesh)
            output["error"] = error

        output["parameters"] = self.parameters
        output["mesh"] = self.mesh

        # add derived quantities to output
        if "derived_quantities" in self.parameters["exports"].keys():
            output["derived_quantities"] = self.derived_quantities_global

        # initialise output["solutions"] with solute and temperature
        output["solutions"] = {
            "solute": res[0],
            "T": self.T
        }
        # add traps to output
        for i in range(len(self.parameters["traps"])):
            output["solutions"]["trap_{}".format(i + 1)] = res[i + 1]
        # compute retention and add it to output
        output["solutions"]["retention"] = project(sum(res), self.V_DG1)
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
