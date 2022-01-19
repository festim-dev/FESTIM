import FESTIM
import fenics as f
import numpy as np
import sympy as sp


class Error(FESTIM.Export):
    def __init__(self, field, exact_solution, norm="error_max", degree=4) -> None:
        super().__init__(field)
        self.exact_solution = exact_solution
        self.norm = norm
        self.degree = degree

    def compute(self, t):
        exact_sol = f.Expression(sp.printing.ccode(self.exact_solution),
            degree=self.degree,
            t=t)

        if self.norm == "error_max":
            mesh = self.function.function_space().mesh()
            vertex_values_u = self.function.compute_vertex_values(mesh)
            vertex_values_sol = exact_sol.compute_vertex_values(mesh)
            error_max = np.max(np.abs(vertex_values_u - vertex_values_sol))
            return error_max
        else:
            error_L2 = f.errornorm(
                exact_sol, self.function, self.norm)
            return error_L2
