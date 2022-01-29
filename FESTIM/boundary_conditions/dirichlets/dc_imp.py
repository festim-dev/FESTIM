from FESTIM import DirichletBC, BoundaryConditionExpression, k_B
import fenics as f
import sympy as sp


def dc_imp(T, phi, R_p, D_0, E_D, Kr_0=None, E_Kr=None):
    D = D_0*f.exp(-E_D/k_B/T)
    value = phi*R_p/D
    if Kr_0 is not None:
        Kr = Kr_0*f.exp(-E_Kr/k_B/T)
        value += (phi/Kr)**0.5

    return value


class ImplantationDirichlet(DirichletBC):
    def __init__(self, surfaces, phi, R_p, D_0, E_D, Kr_0=None, E_Kr=None) -> None:
        super().__init__(surfaces, component=0)
        self.phi = phi
        self.R_p = R_p
        self.D_0 = D_0
        self.E_D = E_D
        self.Kr_0 = Kr_0
        self.E_Kr = E_Kr

    def create_expression(self, T):
        phi = f.Expression(
            sp.printing.ccode(self.phi),
            t=0, degree=1)
        R_p = f.Expression(
            sp.printing.ccode(self.R_p),
            t=0, degree=1)
        sub_expressions = [phi, R_p]
        # D = self.D_0*f.exp(-self.E_D/k_B/T)
        # value = flux_expr*R_p_expr/D

        value_BC = BoundaryConditionExpression(
            T, dc_imp,
            phi=phi, R_p=R_p, D_0=self.D_0, E_D=self.E_D,
            Kr_0=self.Kr_0, E_Kr=self.E_Kr,
            )
        self.expression = value_BC
        self.sub_expressions = sub_expressions
