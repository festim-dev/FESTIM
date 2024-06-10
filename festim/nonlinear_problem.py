import fenics as f


class Problem(f.NonlinearProblem):
    """
    Class to set up a nonlinear variational problem (F(u, v)=0) to solve
    by the Newton method based on the form of the variational problem, the Jacobian
    form of the variational problem, and the boundary conditions

    Args:
        J (ufl.Form): the Jacobian form of the variational problem
        F (ufl.Form): the form of the variational problem
        bcs (list): list of fenics.DirichletBC
    """

    def __init__(self, J, F, bcs):
        self.jacobian_form = J
        self.residual_form = F
        self.bcs = bcs
        self.assembler = f.SystemAssembler(
            self.jacobian_form, self.residual_form, self.bcs
        )
        f.NonlinearProblem.__init__(self)

    def F(self, b, x):
        """Assembles the RHS in Ax=b and applies the boundary conditions"""
        self.assembler.assemble(b, x)

    def J(self, A, x):
        """Assembles the LHS in Ax=b and applies the boundary conditions"""
        self.assembler.assemble(A)
