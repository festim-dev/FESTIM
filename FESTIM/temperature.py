import FESTIM
import sympy as sp
from fenics import *


class Temperature:
    def __init__(self, type, value=None, initial_value=None) -> None:
        self.type = type
        self.T = None
        self.T_n = None
        self.v_T = None
        self.value = value
        self.expression = value
        self.initial_value = initial_value
        self.sub_expressions = []
        self.F = 0
        self.sources = []
        self.boundary_conditions = []

    def create_functions(self, V, materials=None, dx=None, ds=None, dt=None):
        # TODO: materials, dx, ds, dt should be optional
        self.T = Function(V, name="T")
        self.T_n = Function(V, name="T_n")
        if self.type == "expression":
            self.expression = Expression(
                sp.printing.ccode(self.expression), t=0, degree=2)
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
        # TODO needs to choose between having Temperature bcs in Temperature or in Simulation.boundary_conditions
        self.dirichlet_bcs = []
        for bc in self.boundary_conditions:
            if bc.type == "dc" and bc.component == "T":
                bc.create_expression(self.T)
                for surf in bc.surfaces:
                    bci = DirichletBC(V, bc.expression, surface_markers, surf)
                    self.dirichlet_bcs.append(bci)
                self.sub_expressions += bc.sub_expressions
                self.sub_expressions.append(bc.expression)

    def define_variational_problem(self, materials, dx, ds, dt):
        """Create a variational form for heat transfer problem

        Arguments:

        Raises:
            NameError: if thermal_cond is not in keys
            NameError: if heat_capacity is not in keys
            NameError: if rho is not in keys

        Returns:
            fenics.Form -- the formulation for heat transfers problem
            list -- contains the fenics.Expression to be updated
        """

        print('Defining variational problem heat transfers')
        T, T_n = self.T, self.T_n
        v_T = self.v_T

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
            self.F += - src*v_T*dx(source.volume)

        # Boundary conditions
        for bc in self.boundary_conditions:
            if bc.type not in FESTIM.helpers.T_bc_types["dc"]:
                bc.create_form_for_flux(self.T, solute=None)

                # TODO: maybe that's not necessary
                self.sub_expressions += bc.sub_expressions

                for surf in bc.surfaces:
                    self.F += -bc.form*self.v_T*ds(surf)
