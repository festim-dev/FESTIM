from fenics import *

mesh = UnitSquareMesh(5, 5)
V = FunctionSpace(mesh, "P", 1)
u = Function(V)
v = TestFunction(V)
T = Expression("t", t=1, degree=1)


class Coeff(UserExpression):
    def __init__(self, T, **kwargs):
        super().__init__(kwargs)
        self._T = T

    def eval_cell(self, value, x, ufc_cell):
        value[0] = 2*self._T(x)*x[0]

    def value_shape(self):
        return ()


coeff = Coeff(T)

sm = MeshFunction("size_t", mesh, 1, 0)
right = CompiledSubDomain('x[0] > 0.75')
right.mark(sm, 2)
bc1 = DirichletBC(V, coeff, sm, 2)
bc2 = DirichletBC(V, 2*coeff, sm, 2)

F = dot(grad(u), grad(v))*dx

set_log_level(30)
for i in range(1, 10):
    T.t = i
    coeff._T = T

    solve(F == 0, u, bc1)
    a = u(0.5, 0.5)
    solve(F == 0, u, bc2)
    b = u(0.5, 0.5)

    print(a, b)
