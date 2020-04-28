from FESTIM import *
import FESTIM
from fenics import *
import sympy as sp
import numpy as np


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
        if bc["type"] == "dc":
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


def apply_fluxes(boundary_conditions, solutions, testfunctions, ds, T):
    ''' Modifies the formulation and adds fluxes based
    on parameters in boundary_conditions
    '''
    expressions = []
    solute = solutions[0]
    test_solute = testfunctions[0]
    F = 0
    k_B = 8.6e-5
    for bc in boundary_conditions:
        if bc["type"] not in helpers.bc_types["dc"]:
            if bc["type"] not in helpers.bc_types["neumann"] and \
               bc["type"] not in helpers.bc_types["robin"]:

                raise NameError(
                    "Unknown boundary condition type : " + bc["type"])
            if bc["type"] == "flux":
                flux = sp.printing.ccode(bc["value"])
                flux = Expression(flux, t=0,
                                  degree=2)
                expressions.append(flux)
            elif bc["type"] == "recomb":
                Kr = bc["Kr_0"]*exp(-bc["E_Kr"]/FESTIM.k_B/T)
                flux = -Kr*solute**bc["order"]

            if type(bc['surface']) is not list:
                surfaces = [bc['surface']]
            else:
                surfaces = bc['surface']
            for surf in surfaces:
                F += -test_solute*flux*ds(surf)
    return F, expressions


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
                        k_B, T))
            value_BC = Expression(sp.printing.ccode(value_BC), t=0,
                                  degree=2)
            print("WARNING: solubility BC. \
                If temperature is type solve_transient\
                     initial temperature will be considered.")
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

        if BC["type"] not in helpers.bc_types["neumann"] and \
           BC["type"] not in helpers.bc_types["robin"] and \
           BC["type"] not in helpers.bc_types["dc"]:

            raise NameError("Unknown boundary condition type : " + bc["type"])
        if type_BC in helpers.bc_types["dc"]:
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
