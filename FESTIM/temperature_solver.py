import FESTIM
import fenics as f
import sympy as sp


class HeatTransferProblem(FESTIM.Temperature):
    def __init__(self, transient=True, initial_value=0.) -> None:
        """Inits HeatTransferProblem

        Args:
            transient (bool, optional): If True, a transient simulation will
                be run. Defaults to True.
            initial_value (sp.Add, float, optional): The initial value.
                Only needed if transient is True. Defaults to 0..
        """
        super().__init__()
        self.transient = transient
        self.initial_value = initial_value

        self.F = 0
        self.v_T = None
        self.sources = []
        self.boundary_conditions = []
        self.sub_expressions = []

    def create_functions(self, V, materials, dx, ds, dt=None):
        """Creates functions self.T, self.T_n and test function self.v_T.
        Solves the steady-state heat transfer problem if self.transient is
        False.

        Args:
            V (fenics.FunctionSpace): the function space of Temperature
            materials (FESTIM.Materials): the materials.
            dx (fenics.Measure): measure for dx.
            ds (fenics.Measure): measure for ds.
            dt (FESTIM.Stepsize, optional): the stepsize. Only needed if
                self.transient is True. Defaults to None.
        """
        # Define variational problem for heat transfers
        self.T = f.Function(V, name="T")
        self.T_n = f.Function(V, name="T_n")
        self.v_T = f.TestFunction(V)

        if self.transient:
            ccode_T_ini = sp.printing.ccode(self.initial_value)
            self.initial_value = f.Expression(ccode_T_ini, degree=2, t=0)
            self.T_n.assign(f.interpolate(self.initial_value, V))

        self.define_variational_problem(materials, dx, ds, dt)
        self.create_dirichlet_bcs(ds.subdomain_data())

        if not self.transient:
            print("Solving stationary heat equation")
            f.solve(self.F == 0, self.T, self.dirichlet_bcs)
            self.T_n.assign(self.T)

    def define_variational_problem(self, materials, dx, ds, dt=None):
        """Create a variational form for heat transfer problem

        Args:
            materials (FESTIM.Materials): the materials.
            dx (fenics.Measure): measure for dx.
            ds (fenics.Measure): measure for ds.
            dt (FESTIM.Stepsize, optional): the stepsize. Only needed if
                self.transient is True. Defaults to None.
        """

        print('Defining variational problem heat transfers')
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
                    self.F += rho*cp*(T-T_n)/dt.value*v_T*dx(vol)
            # Diffusion term
            for vol in subdomains:
                self.F += f.dot(thermal_cond*f.grad(T), f.grad(v_T))*dx(vol)

        # source term
        for source in self.sources:
            src = sp.printing.ccode(source.value)
            src = f.Expression(src, degree=2, t=0)
            self.sub_expressions.append(src)
            if type(source.volume) is list:
                volumes = source.volume
            else:
                volumes = [source.volume]
            for volume in volumes:
                self.F += - src*v_T*dx(volume)

        # Boundary conditions
        for bc in self.boundary_conditions:
            if not isinstance(bc, FESTIM.DirichletBC):
                bc.create_form(self.T, solute=None)

                # TODO: maybe that's not necessary
                self.sub_expressions += bc.sub_expressions

                for surf in bc.surfaces:
                    self.F += -bc.form*self.v_T*ds(surf)

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
            if isinstance(bc, FESTIM.DirichletBC) and bc.component == "T":
                bc.create_expression(self.T)
                for surf in bc.surfaces:
                    bci = f.DirichletBC(
                        V, bc.expression, surface_markers, surf)
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
            FESTIM.update_expressions(self.sub_expressions, t)
            # Solve heat transfers
            dT = f.TrialFunction(self.T.function_space())
            JT = f.derivative(self.F, self.T, dT)  # Define the Jacobian
            problem = f.NonlinearVariationalProblem(
                self.F, self.T, self.dirichlet_bcs, JT)
            solver = f.NonlinearVariationalSolver(problem)
            newton_solver_prm = solver.parameters["newton_solver"]
            newton_solver_prm["absolute_tolerance"] = 1e-3
            newton_solver_prm["relative_tolerance"] = 1e-10
            solver.solve()
            self.T_n.assign(self.T)
