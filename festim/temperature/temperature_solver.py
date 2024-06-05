import festim
import fenics as f
import sympy as sp
import warnings


class HeatTransferProblem(festim.Temperature):
    """
    Args:
        transient (bool, optional): If True, a transient simulation will
            be run. Defaults to True.
        initial_condition (int, float, sp.Expr, festim.InitialCondition, optional): The initial condition.
            Only needed if transient is True.
        absolute_tolerance (float, optional): the absolute tolerance of the newton
            solver. Defaults to 1e-03
        relative_tolerance (float, optional): the relative tolerance of the newton
            solver. Defaults to 1e-10
        maximum_iterations (int, optional): maximum iterations allowed for
            the solver to converge. Defaults to 30.
        linear_solver (str, optional): linear solver method for the newton solver,
            options can be viewed with print(list_linear_solver_methods()).
            If None, the default fenics linear solver will be used ("umfpack").
            More information can be found at: https://fenicsproject.org/pub/tutorial/html/._ftut1017.html.
            Defaults to None.
        preconditioner (str, optional): preconditioning method for the newton solver,
            options can be veiwed by print(list_krylov_solver_preconditioners()).
            Defaults to "default".

    Attributes:
        F (fenics.Form): the variational form of the heat transfer problem
        v_T (fenics.TestFunction): the test function
        newton_solver (fenics.NewtonSolver): Newton solver for solving the nonlinear problem
        initial_condition (festim.InitialCondition): the initial condition
        sub_expressions (list): contains time dependent fenics.Expression to
            be updated
        sources (list): contains festim.Source objects for volumetric heat
            sources
        boundary_conditions (list): contains festim.BoundaryConditions
    """

    def __init__(
        self,
        transient=True,
        initial_condition=None,
        absolute_tolerance=1e-3,
        relative_tolerance=1e-10,
        maximum_iterations=30,
        linear_solver=None,
        preconditioner="default",
    ) -> None:
        super().__init__()
        self.transient = transient
        self.initial_condition = initial_condition
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.maximum_iterations = maximum_iterations
        self.linear_solver = linear_solver
        self.preconditioner = preconditioner

        self.F = 0
        self.v_T = None
        self.sources = []
        self.boundary_conditions = []
        self.sub_expressions = []
        self.newton_solver = None

    @property
    def newton_solver(self):
        return self._newton_solver

    @newton_solver.setter
    def newton_solver(self, value):
        if value is None:
            self._newton_solver = value
        elif isinstance(value, f.NewtonSolver):
            if self._newton_solver:
                print("Settings for the Newton solver will be overwritten")
            self._newton_solver = value
        else:
            raise TypeError("accepted type for newton_solver is fenics.NewtonSolver")

    @property
    def initial_condition(self):
        return self._initial_condition

    @initial_condition.setter
    def initial_condition(self, value):
        if isinstance(value, (int, float, sp.Expr)):
            self._initial_condition = festim.InitialCondition(field="T", value=value)
        else:
            self._initial_condition = value

    # TODO rename initialise?
    def create_functions(self, materials, mesh, dt=None):
        """Creates functions self.T, self.T_n and test function self.v_T.
        Solves the steady-state heat transfer problem if self.transient is
        False.

        Args:
            materials (festim.Materials): the materials.
            mesh (festim.Mesh): the mesh
            dt (festim.Stepsize, optional): the stepsize. Only needed if
                self.transient is True. Defaults to None.
        """
        # Define variational problem for heat transfers
        V = f.FunctionSpace(mesh.mesh, "CG", 1)
        self.T = f.Function(V, name="T")
        self.T_n = f.Function(V, name="T_n")
        self.v_T = f.TestFunction(V)
        if self.transient and self.initial_condition is None:
            raise AttributeError(
                "Initial condition is required for transient heat transfer simulations"
            )
        if self.transient and self.initial_condition:
            if isinstance(self.initial_condition.value, str):
                if self.initial_condition.value.endswith(".xdmf"):
                    with f.XDMFFile(self.initial_condition.value) as file:
                        file.read_checkpoint(
                            self.T_n,
                            self.initial_condition.label,
                            self.initial_condition.time_step,
                        )
            else:
                ccode_T_ini = sp.printing.ccode(self.initial_condition.value)
                self.initial_condition.value = f.Expression(ccode_T_ini, degree=2, t=0)
                self.T_n.assign(f.interpolate(self.initial_condition.value, V))

        self.define_variational_problem(materials, mesh, dt)
        self.create_dirichlet_bcs(mesh.surface_markers)

        if not self.newton_solver:
            self.define_newton_solver()

        if not self.transient:
            print("Solving stationary heat equation")
            dT = f.TrialFunction(self.T.function_space())
            JT = f.derivative(self.F, self.T, dT)
            problem = festim.Problem(JT, self.F, self.dirichlet_bcs)

            f.begin(
                "Solving nonlinear variational problem."
            )  # Add message to fenics logs
            self.newton_solver.solve(problem, self.T.vector())
            f.end()

            self.T_n.assign(self.T)

    def define_variational_problem(self, materials, mesh, dt=None):
        """Create a variational form for heat transfer problem

        Args:
            materials (festim.Materials): the materials.
            mesh (festim.Mesh): the mesh.
            dt (festim.Stepsize, optional): the stepsize. Only needed if
                self.transient is True. Defaults to None.
        """

        print("Defining variational problem heat transfers")
        T, T_n = self.T, self.T_n
        v_T = self.v_T

        self.F = 0
        for mat in materials:
            thermal_cond = mat.thermal_cond
            if callable(thermal_cond):  # if thermal_cond is a function
                thermal_cond = thermal_cond(T)

            subdomains = mat.id  # list of subdomains with this material
            if type(subdomains) is not list:
                subdomains = [subdomains]  # make sure subdomains is a list
            if self.transient:
                cp = mat.heat_capacity
                rho = mat.rho
                if callable(cp):  # if cp or rho are functions, apply T
                    cp = cp(T)
                if callable(rho):
                    rho = rho(T)
                # Transien term
                for vol in subdomains:
                    self.F += rho * cp * (T - T_n) / dt.value * v_T * mesh.dx(vol)
            # Diffusion term
            for vol in subdomains:
                if mesh.type == "cartesian":
                    self.F += f.dot(thermal_cond * f.grad(T), f.grad(v_T)) * mesh.dx(
                        vol
                    )
                elif mesh.type == "cylindrical":
                    r = f.SpatialCoordinate(mesh.mesh)[0]
                    self.F += (
                        r
                        * f.dot(thermal_cond * f.grad(T), f.grad(v_T / r))
                        * mesh.dx(vol)
                    )
                elif mesh.type == "spherical":
                    r = f.SpatialCoordinate(mesh.mesh)[0]
                    self.F += (
                        thermal_cond
                        * r
                        * r
                        * f.dot(f.grad(T), f.grad(v_T / r / r))
                        * mesh.dx(vol)
                    )
        # source term
        for source in self.sources:
            self.sub_expressions.append(source.value)
            if type(source.volume) is list:
                volumes = source.volume
            else:
                volumes = [source.volume]
            for volume in volumes:
                self.F += -source.value * v_T * mesh.dx(volume)

        # Boundary conditions
        for bc in self.boundary_conditions:
            if isinstance(bc, festim.FluxBC):
                bc.create_form(self.T, solute=None)

                # TODO: maybe that's not necessary
                self.sub_expressions += bc.sub_expressions

                for surf in bc.surfaces:
                    self.F += -bc.form * self.v_T * mesh.ds(surf)

    def define_newton_solver(self):
        """Creates the Newton solver and sets its parameters"""
        self.newton_solver = f.NewtonSolver(f.MPI.comm_world)
        self.newton_solver.parameters["error_on_nonconvergence"] = False
        self.newton_solver.parameters["absolute_tolerance"] = self.absolute_tolerance
        self.newton_solver.parameters["relative_tolerance"] = self.relative_tolerance
        self.newton_solver.parameters["maximum_iterations"] = self.maximum_iterations
        self.newton_solver.parameters["linear_solver"] = self.linear_solver
        self.newton_solver.parameters["preconditioner"] = self.preconditioner

    def create_dirichlet_bcs(self, surface_markers):
        """Creates a list of fenics.DirichletBC and add time dependent
        expressions to .sub_expressions

        Args:
            surface_markers (fenics.MeshFunction): contains the mesh facet
                markers
        """
        V = self.T.function_space()
        self.dirichlet_bcs = []
        for bc in self.boundary_conditions:
            if isinstance(bc, festim.DirichletBC) and bc.field == "T":
                bc.create_expression(self.T)
                for surf in bc.surfaces:
                    bci = f.DirichletBC(V, bc.expression, surface_markers, surf)
                    self.dirichlet_bcs.append(bci)
                self.sub_expressions += bc.sub_expressions
                self.sub_expressions.append(bc.expression)

    def update(self, t):
        """Updates T_n, and T with respect to time by solving the heat transfer
        problem

        Args:
            t (float): the time
        """
        if self.transient:
            festim.update_expressions(self.sub_expressions, t)
            # Solve heat transfers
            dT = f.TrialFunction(self.T.function_space())
            JT = f.derivative(self.F, self.T, dT)  # Define the Jacobian
            problem = festim.Problem(JT, self.F, self.dirichlet_bcs)

            f.begin(
                "Solving nonlinear variational problem."
            )  # Add message to fenics logs
            self.newton_solver.solve(problem, self.T.vector())
            f.end()

            self.T_n.assign(self.T)

    def is_steady_state(self):
        return not self.transient
