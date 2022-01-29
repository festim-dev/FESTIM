import FESTIM
import sympy as sp
from fenics import *


class Temperature:
    """
    Description of Temperature

    Attributes:
        type (str): the type of temperature
        T (fenics.Function): the function attributed with temperature
        T_n (fenics.Function): the previous function
        v_T (fenics.TestFunction): the test function
        value (sp.Add, int, float): the expression of temperature
        expression (fenics.Expression): the expression of temperature as a
            fenics object
        initial_value (sp.Add, int, float): the initial value
        sub_expressions (list): contains time dependent fenics.Expression to
            be updated
        F (fenics.Form): the variational form of the heat transfer problem
        sources (list): contains FESTIM.Source objects for volumetric heat
            sources
        boundary_conditions (list): contains FESTIM.BoundaryConditions
    """
    def __init__(self, type, value=None, initial_value=None) -> None:
        """Inits Temperature

        Args:
            type (str): type of temperature in "expression",
                "solve_stationary", "solve_transient"
            value (sp.Add, int, float, optional): The value of the temperature.
                Only needed if type is not "expression". Defaults to None.
            initial_value (sp.Add, int, float, optional): The initial value.
                Only needed if type is not "expression". Defaults to None.
        """
        self.type = type
        self.T = None
        self.T_n = None
        self.v_T = None
        self.value = value
        self.initial_value = initial_value
        self.expression = None
        self.sub_expressions = []
        self.F = 0
        self.sources = []
        self.boundary_conditions = []

    def create_functions(self, V, materials=None, dx=None, ds=None, dt=None):
        """Creates functions self.T, self.T_n and test function self.v_T

        Args:
            V (fenics.FunctionSpace): the function space of Temperature
            materials (FESTIM.Materials, optional): needed if self.type is not
                "expression". Defaults to None.
            dx (fenics.Measure, optional): measure for dx. Needed if type is
                not "expression". Defaults to None.
            ds (fenics.Measure, optional): measure for ds. Needed if type is
                not "expression". Defaults to None.
            dt (FESTIM.Stepsize, optional): the stepsize. Needed if type is
                not "expression". Defaults to None.
        """
        # TODO: materials, dx, ds, dt should be optional
        self.T = Function(V, name="T")
        self.T_n = Function(V, name="T_n")
        if self.type == "expression":
            self.expression = Expression(
                sp.printing.ccode(self.value), t=0, degree=2)
            self.T.assign(interpolate(self.expression, V))
            self.T_n.assign(self.T)
        else:
            # Define variational problem for heat transfers
            self.v_T = TestFunction(V)

            if self.type == "solve_transient":
                ccode_T_ini = sp.printing.ccode(self.initial_value)
                self.initial_value = Expression(ccode_T_ini, degree=2, t=0)
                self.T_n.assign(interpolate(self.initial_value, V))

            self.define_variational_problem(materials, dx, ds, dt)
            # self.expressions += expressions_bcs_T + self.expressions_FT
            self.create_dirichlet_bcs(V, ds.subdomain_data())

            if self.type == "solve_stationary":
                print("Solving stationary heat equation")
                solve(self.F == 0, self.T, self.dirichlet_bcs)
                self.T_n.assign(self.T)

    def create_dirichlet_bcs(self, V, surface_markers):
        """Creates a list of fenics.DirichletBC and add time dependent
        expressions to .sub_expressions

        Args:
            V (fenics.FunctionSpace): the function space
            surface_markers (fenics.MeshFunction): contains the mesh facet
                markers
        """
        self.dirichlet_bcs = []
        for bc in self.boundary_conditions:
            if isinstance(bc, FESTIM.DirichletBC) and bc.component == "T":
                bc.create_expression(self.T)
                for surf in bc.surfaces:
                    bci = DirichletBC(V, bc.expression, surface_markers, surf)
                    self.dirichlet_bcs.append(bci)
                self.sub_expressions += bc.sub_expressions
                self.sub_expressions.append(bc.expression)

    def define_variational_problem(self, materials, dx, ds, dt):
        """Create a variational form for heat transfer problem

        Args:
            materials (FESTIM.Materials, optional): the materials. Defaults to
                None.
            dx (fenics.Measure, optional): measure for dx. Defaults to None.
            ds (fenics.Measure, optional): measure for ds. Defaults to None.
            dt (FESTIM.Stepsize, optional): the stepsize. Defaults to None.
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
            if self.type == "solve_transient":
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
                self.F += dot(thermal_cond*grad(T), grad(v_T))*dx(vol)

        # source term
        for source in self.sources:
            src = sp.printing.ccode(source.value)
            src = Expression(src, degree=2, t=0)
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
