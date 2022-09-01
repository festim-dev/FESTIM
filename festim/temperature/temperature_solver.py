import festim
import fenics as f
import sympy as sp


class HeatTransferProblem(festim.Temperature):
    """
    Args:
        transient (bool, optional): If True, a transient simulation will
            be run. Defaults to True.
        initial_value (sp.Add, float, optional): The initial value.
            Only needed if transient is True. Defaults to 0.
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

    Attributes:
        F (fenics.Form): the variational form of the heat transfer problem
        v_T (fenics.TestFunction): the test function
        initial_value (sp.Add, int, float): the initial value
        sub_expressions (list): contains time dependent fenics.Expression to
            be updated
        sources (list): contains festim.Source objects for volumetric heat
            sources
        boundary_conditions (list): contains festim.BoundaryConditions
    """

    def __init__(
        self,
        transient=True,
        initial_value=0.0,
        absolute_tolerance=1e-3,
        relative_tolerance=1e-10,
        maximum_iterations=30,
        linear_solver=None,
    ) -> None:
        super().__init__()
        self.transient = transient
        self.initial_value = initial_value
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.maximum_iterations = maximum_iterations
        self.linear_solver = linear_solver

        self.F = 0
        self.v_T = None
        self.sources = []
        self.boundary_conditions = []
        self.sub_expressions = []

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

        if self.transient:
            ccode_T_ini = sp.printing.ccode(self.initial_value)
            self.initial_value = f.Expression(ccode_T_ini, degree=2, t=0)
            self.T_n.assign(f.interpolate(self.initial_value, V))

        self.define_variational_problem(materials, mesh, dt)
        self.create_dirichlet_bcs(mesh.surface_markers)

        if not self.transient:
            print("Solving stationary heat equation")
            dT = f.TrialFunction(self.T.function_space())
            JT = f.derivative(self.F, self.T, dT)
            problem = f.NonlinearVariationalProblem(
                self.F, self.T, self.dirichlet_bcs, JT
            )
            solver = f.NonlinearVariationalSolver(problem)
            newton_solver_prm = solver.parameters["newton_solver"]
            newton_solver_prm["absolute_tolerance"] = self.absolute_tolerance
            newton_solver_prm["relative_tolerance"] = self.relative_tolerance
            newton_solver_prm["maximum_iterations"] = self.maximum_iterations
            newton_solver_prm["linear_solver"] = self.linear_solver
            solver.solve()
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
        for mat in materials.materials:
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
            problem = f.NonlinearVariationalProblem(
                self.F, self.T, self.dirichlet_bcs, JT
            )
            solver = f.NonlinearVariationalSolver(problem)
            newton_solver_prm = solver.parameters["newton_solver"]
            newton_solver_prm["absolute_tolerance"] = self.absolute_tolerance
            newton_solver_prm["relative_tolerance"] = self.relative_tolerance
            newton_solver_prm["maximum_iterations"] = self.maximum_iterations
            newton_solver_prm["linear_solver"] = self.linear_solver
            solver.solve()
            self.T_n.assign(self.T)

    def is_steady_state(self):
        return not self.transient
