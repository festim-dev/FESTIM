from fenics import *
import festim


class HTransportProblem:
    """Hydrogen Transport Problem.
    Used internally in festim.Simulation

    Args:
        mobile (festim.Mobile): the mobile concentration
        traps (festim.Traps): the traps
        T (festim.Temperature): the temperature
        settings (festim.Settings): the problem settings
        initial_conditions (list of festim.initial_conditions): the
            initial conditions of the h transport problem

    Attributes:
        expressions (list): contains time-dependent fenics.Expressions
        J (ufl.Form): the jacobian of the variational problem
        V (fenics.FunctionSpace): the vector-function space for concentrations
        u (fenics.Function): the vector holding the concentrations (c_m, ct1,
            ct2, ...)
        v (fenics.TestFunction): the test function
        u_n (fenics.Function): the "previous" function
        newton_solver (fenics.NewtonSolver): Newton solver for solving the nonlinear problem
        bcs (list): list of fenics.DirichletBC for H transport
    """

    def __init__(self, mobile, traps, T, settings, initial_conditions) -> None:
        self.mobile = mobile
        self.traps = traps
        self.T = T
        self.settings = settings
        self.initial_conditions = initial_conditions

        self.J = None
        self.u = None
        self.v = None
        self.u_n = None
        self.newton_solver = None

        self.boundary_conditions = []
        self.bcs = None
        self.V = None
        self.V_CG1 = None
        self.expressions = []

    @property
    def newton_solver(self):
        return self._newton_solver

    @newton_solver.setter
    def newton_solver(self, value):
        if value is None:
            self._newton_solver = value
        elif isinstance(value, NewtonSolver):
            if self._newton_solver:
                print("Settings for the Newton solver will be overwritten")
            self._newton_solver = value
        else:
            raise TypeError("accepted type for newton_solver is fenics.NewtonSolver")

    @property
    def _all_surf_kinetics(self):
        return [
            bc
            for bc in self.boundary_conditions
            if isinstance(bc, festim.SurfaceKinetics)
        ]

    def initialise(self, mesh, materials, dt=None):
        """Assigns BCs, create suitable function space, initialise
        concentration fields, define variational problem

        Args:
            mesh (festim.Mesh): the mesh
            materials (festim.Materials): the materials
            dt (festim.Stepsize, optional): the stepsize, only needed if
                self.settings.transient is True. Defaults to None.
        """
        if self.settings.chemical_pot:
            self.mobile.S = materials.S
            self.mobile.materials = materials
            self.mobile.volume_markers = mesh.volume_markers
            self.mobile.T = self.T
        self.attribute_flux_boundary_conditions()

        self.traps.assign_traps_ids()

        # Define functions
        self.define_function_space(mesh)
        self.initialise_concentrations()
        self.traps.make_traps_materials(materials)
        self.traps.initialise_extrinsic_traps(self.V_CG1)

        # Define variational problem H transport
        # if chemical pot create form to convert theta to concentration
        if self.settings.chemical_pot:
            self.mobile.create_form_post_processing(self.V_DG1, materials, mesh.dx)

        self.define_variational_problem(materials, mesh, dt)
        self.define_newton_solver()

        # Boundary conditions
        print("Defining boundary conditions")
        self.create_dirichlet_bcs(materials, mesh)
        if self.settings.transient:
            self.traps.define_variational_problem_extrinsic_traps(mesh.dx, dt, self.T)
            self.traps.define_newton_solver_extrinsic_traps()

    def define_function_space(self, mesh):
        """Creates a suitable function space for H transport problem

        Args:
            mesh (festim.Mesh): the mesh
        """
        order_trap = 1
        element_solute, order_solute = "CG", 1

        # function space for H concentrations
        nb_traps = len(self.traps)

        # the number of surfaces where SurfaceKinetics is used
        nb_adsorbed = sum([len(bc.surfaces) for bc in self._all_surf_kinetics])

        if nb_traps == 0 and nb_adsorbed == 0:
            V = FunctionSpace(mesh.mesh, element_solute, order_solute)
        else:
            solute = FiniteElement(element_solute, mesh.mesh.ufl_cell(), order_solute)
            traps = FiniteElement(
                self.settings.traps_element_type, mesh.mesh.ufl_cell(), order_trap
            )
            adsorbed = FiniteElement("R", mesh.mesh.ufl_cell(), 0)
            element = [solute] + [traps] * nb_traps + [adsorbed] * nb_adsorbed
            V = FunctionSpace(mesh.mesh, MixedElement(element))

        self.V = V
        self.V_CG1 = FunctionSpace(mesh.mesh, "CG", 1)
        self.V_DG1 = FunctionSpace(mesh.mesh, "DG", 1)

    def initialise_concentrations(self):
        """Creates the main fenics.Function (holding all the concentrations),
        eventually split it and assign it to Trap and Mobile.
        Then initialise self.u_n based on self.initial_conditions

        Args:
            materials (festim.Materials): the materials
        """
        # TODO rename u and u_n to c and c_n
        self.u = Function(self.V, name="c")  # Function for concentrations
        self.v = TestFunction(self.V)  # TestFunction for concentrations
        self.u_n = Function(self.V, name="c_n")

        if self.V.num_sub_spaces() == 0:
            self.mobile.solution = self.u
            self.mobile.previous_solution = self.u_n
            self.mobile.test_function = self.v
        else:
            conc_list = [self.mobile]
            if self.traps:
                conc_list += [*self.traps]
            if len(self._all_surf_kinetics) > 0:
                conc_list += self._all_surf_kinetics

            index = 0
            for concentration in conc_list:
                if isinstance(concentration, festim.SurfaceKinetics):
                    # iterate through each surface of each SurfaceKinetics
                    for i in range(len(concentration.surfaces)):
                        concentration.solutions[i] = self.u.sub(index)
                        concentration.previous_solutions[i] = self.u_n.sub(index)
                        concentration.test_functions[i] = list(split(self.v))[index]
                        index += 1
                else:
                    concentration.solution = self.u.sub(index)
                    concentration.previous_solution = self.u_n.sub(index)
                    concentration.test_function = list(split(self.v))[index]
                    index += 1

        print("Defining initial values")
        field_to_component = {
            "solute": 0,
            "0": 0,
            0: 0,
        }
        for i, trap in enumerate(self.traps, 1):
            field_to_component[trap.id] = i
            field_to_component[str(trap.id)] = i
        # TODO refactore this, attach the initial conditions to the objects directly
        for ini in self.initial_conditions:
            value = ini.value
            component = field_to_component[ini.field]

            if self.V.num_sub_spaces() == 0:
                functionspace = self.V
            else:
                functionspace = self.V.sub(component).collapse()

            if component == 0:
                self.mobile.initialise(
                    functionspace, value, label=ini.label, time_step=ini.time_step
                )
            else:
                trap = self.traps.get_trap(component)
                trap.initialise(
                    functionspace, value, label=ini.label, time_step=ini.time_step
                )

        # assign initial condition for SurfaceKinetics BC
        # iterate through each surface of each SurfaceKinetics
        index = len(self.traps) + 1
        for bc in self._all_surf_kinetics:
            for i in range(len(bc.previous_solutions)):
                functionspace = self.V.sub(index).collapse()
                comp = interpolate(Constant(bc.initial_condition), functionspace)
                assign(bc.previous_solutions[i], comp)
                index += 1

        # initial guess needs to be non zero if chemical pot
        if self.settings.chemical_pot:
            if self.V.num_sub_spaces() == 0:
                functionspace = self.V
            else:
                functionspace = self.V.sub(0).collapse()
            initial_guess = project(
                self.mobile.previous_solution + Constant(DOLFIN_EPS), functionspace
            )
            self.mobile.solution.assign(initial_guess)
        # this is needed to correctly create the formulation
        # TODO: write a test for this?
        if self.V.num_sub_spaces() != 0:
            index = 0
            for concentration in conc_list:
                if isinstance(concentration, festim.SurfaceKinetics):
                    for i in range(len(concentration.surfaces)):
                        concentration.solutions[i] = list(split(self.u))[index]
                        concentration.previous_solutions[i] = list(split(self.u_n))[
                            index
                        ]
                        index += 1
                else:
                    concentration.solution = list(split(self.u))[index]
                    concentration.previous_solution = list(split(self.u_n))[index]
                    index += 1

    def define_variational_problem(self, materials, mesh, dt=None):
        """Creates the variational problem for hydrogen transport (form,
        Dirichlet boundary conditions)

        Args:
            materials (festim.Materials): the materials
            mesh (festim.Mesh): the mesh
            dt (festim.Stepsize, optional): the stepsize, only needed if
                self.settings.transient is True. Defaults to None.
        """
        print("Defining variational problem")
        expressions = []
        F = 0

        # diffusion + transient terms

        self.mobile.create_form(
            materials, mesh, self.T, dt, traps=self.traps, soret=self.settings.soret
        )
        F += self.mobile.F
        expressions += self.mobile.sub_expressions

        # Add traps
        self.traps.create_forms(self.mobile, materials, self.T, mesh.dx, dt)
        F += self.traps.F
        expressions += self.traps.sub_expressions
        self.F = F
        self.expressions = expressions

    def define_newton_solver(self):
        """Creates the Newton solver and sets its parameters"""
        self.newton_solver = NewtonSolver(MPI.comm_world)
        self.newton_solver.parameters["error_on_nonconvergence"] = False
        self.newton_solver.parameters["absolute_tolerance"] = (
            self.settings.absolute_tolerance
        )
        self.newton_solver.parameters["relative_tolerance"] = (
            self.settings.relative_tolerance
        )
        self.newton_solver.parameters["maximum_iterations"] = (
            self.settings.maximum_iterations
        )
        self.newton_solver.parameters["linear_solver"] = self.settings.linear_solver
        self.newton_solver.parameters["preconditioner"] = self.settings.preconditioner

    def attribute_flux_boundary_conditions(self):
        """Iterates through self.boundary_conditions, checks if it's a FluxBC
        and its field is 0, and assign fluxes to self.mobile
        """
        for bc in self.boundary_conditions:
            if isinstance(bc, festim.FluxBC) and bc.field == 0:
                self.mobile.boundary_conditions.append(bc)

    def create_dirichlet_bcs(self, materials, mesh):
        """Creates fenics.DirichletBC objects for the hydrogen transport
        problem and add them to self.bcs
        """
        self.bcs = []
        for bc in self.boundary_conditions:
            if bc.field != "T" and isinstance(bc, festim.DirichletBC):
                bc.create_dirichletbc(
                    self.V,
                    self.T.T,
                    mesh.surface_markers,
                    chemical_pot=self.settings.chemical_pot,
                    materials=materials,
                    volume_markers=mesh.volume_markers,
                )
                self.bcs += bc.dirichlet_bc
                self.expressions += bc.sub_expressions
                self.expressions.append(bc.expression)

    def compute_jacobian(self):
        du = TrialFunction(self.u.function_space())
        self.J = derivative(self.F, self.u, du)

    def update(self, t, dt):
        """Updates the H transport problem.

        Args:
            t (float): the current time (s)
            dt (festim.Stepsize): the stepsize
        """
        festim.update_expressions(self.expressions, t)

        converged = False
        u_ = Function(self.u.function_space())
        u_.assign(self.u)
        while converged is False:
            self.u.assign(u_)
            nb_it, converged = self.solve_once()
            if dt.adaptive_stepsize is not None or dt.milestones is not None:
                dt.adapt(t, nb_it, converged)

        # Update previous solutions
        self.update_previous_solutions()

        # Solve extrinsic traps formulation
        self.traps.solve_extrinsic_traps()

    def solve_once(self):
        """Solves non linear problem

        Returns:
            int, bool: number of iterations for reaching convergence, True if
                converged else False
        """
        if self.J is None:  # Define the Jacobian
            du = TrialFunction(self.u.function_space())
            J = derivative(self.F, self.u, du)
        else:
            J = self.J
        problem = festim.Problem(J, self.F, self.bcs)

        begin("Solving nonlinear variational problem.")  # Add message to fenics logs
        nb_it, converged = self.newton_solver.solve(problem, self.u.vector())
        end()

        return nb_it, converged

    def update_previous_solutions(self):
        self.u_n.assign(self.u)
        self.traps.update_extrinsic_traps_density()

    def update_post_processing_solutions(self, exports):
        if self.u.function_space().num_sub_spaces() == 0:
            res = [self.u]
        else:
            res = list(self.u.split())

        for i, trap in enumerate(self.traps, 1):
            trap.post_processing_solution = res[i]

        index = len(self.traps) + 1
        for bc in self._all_surf_kinetics:
            for i in range(len(bc.post_processing_solutions)):
                bc.post_processing_solutions[i] = res[index]
                index += 1

        if self.settings.chemical_pot:
            self.mobile.post_processing_solution_to_concentration()
        else:
            self.mobile.post_processing_solution = res[0]
