from FESTIM import *
from fenics import *
import sympy as sp

class ExpressionFromInterpolatedData(UserExpression):
    def __init__(self, t, fun, **kwargs):
        self.t = t
        self._fun = fun
        UserExpression.__init__(self, **kwargs)

    def eval(self, values, x):
        values[:] = float(self._fun(self.t))


def define_dirichlet_bcs_T(parameters, V, boundaries):
    '''
    Arguments:
    - parameters: dict, contains materials and temperature parameters
    - V: FEniCS FunctionSpace(), functionspace of temperature
    - boundaries: FEniCS MeshFunction(), markers for facets.
    Returns:
    - bcs: list, contains FEniCS DirichletBC()
    - expressions: list, contains FEniCS Expression() to be updated
    '''
    bcs = []
    expressions = []
    for bc in parameters["temperature"]["boundary_conditions"]:
        if bc["type"] == "dirichlet":
            value = sp.printing.ccode(bc["value"])
            value = Expression(value, degree=2, t=0)
            expressions.append(value)
            if type(bc["surface"]) is list:
                surfaces = bc["surface"]
            else:
                surfaces = [bc["surface"]]
            for surf in surfaces:
                bci = DirichletBC(V, value, boundaries, surf)
                bcs.append(bci)
    return bcs, expressions


def solubility(S_0, E_S, k_B, T):
    return S_0*exp(-E_S/k_B/T)


def solubility_BC(P, S):
    return P**0.5*S


def apply_boundary_conditions(boundary_conditions, V,
                              surface_marker, ds, temp):
    '''
    Create a list of DirichletBCs.
    Arguments:
    - boundary_conditions: list, parameters for bcs
    - V: FunctionSpace,
    - surface_marker: MeshFunction, contains the markers for
    the different surfaces
    - ds: Measurement
    - temp: Expression, temperature.
    Returns:
    - bcs: list, contains fenics DirichletBC
    - expression: list, contains the fenics Expression
    to be updated.
    '''
    bcs = list()
    expressions = list()
    for BC in boundary_conditions:
        try:
            type_BC = BC["type"]
        except:
            raise KeyError("Missing boundary condition type key")
        if type_BC == "dc":
            value_BC = sp.printing.ccode(BC['value'])
            value_BC = Expression(value_BC, t=0, degree=4)
        elif type_BC == "solubility":
            pressure = BC["pressure"]
            value_BC = solubility_BC(
                    pressure, BC["density"]*solubility(
                        BC["S_0"], BC["E_S"],
                        k_B, T(0)))
            value_BC = Expression(sp.printing.ccode(value_BC), t=0,
                                  degree=2)
        elif type_BC == "table":
            table = BC["value"]
            # Interpolate table
            interpolant = interp1d(
                list(np.array(table)[:, 0]),
                list(np.array(table)[:, 1]),
                fill_value='extrapolate')
            # create UserExpression based on interpolant and t
            value_BC = ExpressionFromInterpolatedData(
                t=0, fun=interpolant, element=V.ufl_element())
        else:
            raise NameError("Unknown boundary condition type")
        expressions.append(value_BC)
        try:
            # Fetch the component of the BC
            component = BC["component"]
        except:
            # By default, component is solute (ie. 0)
            component = 0
        if type(BC['surface']) is not list:
            surfaces = [BC['surface']]
        else:
            surfaces = BC['surface']
        if V.num_sub_spaces() == 0:
            funspace = V
        else:  # if only one component, use subspace
            funspace = V.sub(component)
        for surface in surfaces:
            bci = DirichletBC(funspace, value_BC,
                              surface_marker, surface)
            bcs.append(bci)

    return bcs, expressions
