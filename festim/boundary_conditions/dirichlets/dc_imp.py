from festim import DirichletBC, BoundaryConditionExpression, k_B
import fenics as f
import sympy as sp


def dc_imp(T, phi, R_p, D_0, E_D, Kr_0=None, E_Kr=None):
    D = D_0 * f.exp(-E_D / k_B / T)
    value = phi * R_p / D
    if Kr_0 is not None:
        Kr = Kr_0 * f.exp(-E_Kr / k_B / T)
        value += (phi / Kr) ** 0.5

    return value


class ImplantationDirichlet(DirichletBC):
    """Subclass of DirichletBC representing an approximation of an implanted
    flux of hydrogen.
    The details of the approximation can be found in
    https://www.nature.com/articles/s41598-020-74844-w

    c = phi*R_p/D + (phi/Kr)**0.5

    Args:
        surfaces (list or int): the surfaces of the BC
        phi (float or sp.Expr): implanted flux (H/m2/s)
        R_p (float or sp.Expr): implantation depth (m)
        D_0 (float): diffusion coefficient pre-exponential factor (m2/s)
        E_D (float): diffusion coefficient activation energy (eV)
        Kr_0 (float, optional): recombination coefficient pre-exponential
            factor (m^4/s). If None, instantaneous recombination will be
            assumed. Defaults to None.
        E_Kr (float, optional): recombination coefficient activation
            energy (eV). Defaults to None.
    """

    def __init__(self, surfaces, phi, R_p, D_0, E_D, Kr_0=None, E_Kr=None) -> None:
        super().__init__(surfaces, field=0, value=None)
        self.phi = phi
        self.R_p = R_p
        self.D_0 = D_0
        self.E_D = E_D
        self.Kr_0 = Kr_0
        self.E_Kr = E_Kr

    def create_expression(self, T):
        phi = f.Expression(sp.printing.ccode(self.phi), t=0, degree=1)
        R_p = f.Expression(sp.printing.ccode(self.R_p), t=0, degree=1)
        sub_expressions = [phi, R_p]

        value_BC = BoundaryConditionExpression(
            T,
            dc_imp,
            phi=phi,
            R_p=R_p,
            D_0=self.D_0,
            E_D=self.E_D,
            Kr_0=self.Kr_0,
            E_Kr=self.E_Kr,
        )
        self.expression = value_BC
        self.sub_expressions = sub_expressions
