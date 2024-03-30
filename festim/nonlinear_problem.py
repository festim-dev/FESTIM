import fenics as f


class Problem(f.NonlinearProblem):
    """
    Class to set up the nonlinear variational problem (F(u, v)=0) to solve
    by the Newton method based on the form of the variational problem, the Jacobian
    form of the variational problem, and the boundary conditions

    Args:
        bilinear_form (ufl.Form): the Jacobian form of the variational problem
        linear_form (ufl.Form): the form of the variational problem
        bcs (list): list of fenics.DirichletBC
    """

    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        f.NonlinearProblem.__init__(self)

    def F(self, b, x):
        """Assembles the RHS in AdU=b (JdU=F) and applies the boundary conditions"""
        f.assemble(self.linear_form, tensor=b)
        if self.bcs:
            for bc in self.bcs:
                bc.apply(b, x)

    def J(self, A, x):
        """Assembles the LHS in AdU=b (JdU=F) and applies the boundary conditions"""
        f.assemble(self.bilinear_form, tensor=A)
        if self.bcs:
            for bc in self.bcs:
                bc.apply(A)
