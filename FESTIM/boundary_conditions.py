import FESTIM
from fenics import *
import sympy as sp
import numpy as np


def define_dirichlet_bcs_T(simulation):
    """Creates a list of BCs for thermal problem

    Arguments:


    Returns:
        list -- contains fenics.DirichletBC
        list -- contains fenics.Expression to be updated
    """
    parameters = simulation.parameters
    V = simulation.V_CG1
    boundaries = simulation.surface_markers

    bcs = []
    expressions = []
    for bc in parameters["temperature"]["boundary_conditions"]:
        if bc["type"] == "dc":
            value = sp.printing.ccode(bc["value"])
            value = Expression(value, degree=2, t=0)
            expressions.append(value)
            if type(bc["surfaces"]) is list:
                surfaces = bc["surfaces"]
            else:
                surfaces = [bc["surfaces"]]
            for surf in surfaces:
                bci = DirichletBC(V, value, boundaries, surf)
                bcs.append(bci)
    return bcs, expressions


def apply_fluxes(simulation):
    """Modifies the formulation and adds fluxes based
    on parameters in boundary_conditions

    Arguments:


    Keyword Arguments:
        S {fenics.UserExpression} -- solubility (default: {None})

    Raises:
        NameError: if boundary condition type is unknown

    Returns:
        fenics.Form() -- formulation for BCs
        list -- contains all the fenics.Expression() to be updated
    """
    parameters = simulation.parameters
    u = simulation.u
    v = simulation.v
    ds = simulation.ds
    T = simulation.T
    S = simulation.S

    expressions = []
    solutions = split(u)
    testfunctions = split(v)
    solute = solutions[0]
    test_solute = testfunctions[0]
    F = 0
    k_B = FESTIM.k_B
    boundary_conditions = parameters["boundary_conditions"]
    conservation_chemic_pot = False
    if S is not None:
        for mat in parameters["materials"]:
            if "S_0" in mat.keys() or "E_S" in mat.keys():
                conservation_chemic_pot = True
        if conservation_chemic_pot:
            solute = solute*S

    for bc in boundary_conditions:
        if bc["type"] not in FESTIM.helpers.bc_types["dc"]:
            if bc["type"] not in FESTIM.helpers.bc_types["neumann"] and \
               bc["type"] not in FESTIM.helpers.bc_types["robin"]:

                raise NameError(
                    "Unknown boundary condition type : " + bc["type"])
            if bc["type"] == "flux":
                flux = sp.printing.ccode(bc["value"])
                flux = Expression(flux, t=0,
                                  degree=2)
                expressions.append(flux)
            elif bc["type"] == "recomb":
                Kr = bc["Kr_0"]*exp(-bc["E_Kr"]/k_B/T)
                flux = -Kr*solute**bc["order"]

            if type(bc['surfaces']) is not list:
                surfaces = [bc['surfaces']]
            else:
                surfaces = bc['surfaces']
            for surf in surfaces:
                F += -test_solute*flux*ds(surf)
    return F, expressions


class BoundaryConditionTheta(UserExpression):
    """Creates an Expression for converting dirichlet bcs in the case
    of chemical potential conservation

    Args:
        UserExpression (fenics.UserExpression):
    """
    def __init__(self, bci, mesh, materials, vm, T, **kwargs):
        """initialisation

        Args:
            bci (fenics.Expression): value of BC
            mesh (fenics.mesh): mesh
            materials (list): contains dicts for materials
            vm (fenics.MeshFunction): volume markers
            T (fenics.Function): Temperature
        """
        super().__init__(kwargs)
        self._bci = bci
        self._mesh = mesh
        self._vm = vm
        self._T = T
        self._materials = materials

    def eval_cell(self, value, x, ufc_cell):
        cell = Cell(self._mesh, ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = FESTIM.helpers.find_material_from_id(
            self._materials, subdomain_id)
        S_0 = material["S_0"]
        E_S = material["E_S"]
        value[0] = self._bci(x)/(S_0*exp(-E_S/FESTIM.k_B/self._T(x)))

    def value_shape(self):
        return ()


class BoundaryCondition(UserExpression):
    def __init__(self, T, prms, eval_function):

        super().__init__()
        # create Expressions or Constant for all parameters
        for key, value in prms.items():
            if isinstance(value, (int, float)):
                prms[key] = Constant(value)
            else:
                prms[key] = Expression(sp.printing.ccode(value),
                                       t=0,
                                       degree=1)

        self.prms = prms
        self._T = T
        self.eval_function = eval_function

    def eval(self, value, x):
        # find local value of parameters
        new_prms = {}
        for key, prm_val in self.prms.items():
            new_prms[key] = prm_val(x)

        # evaluate at local point
        value[0] = self.eval_function(self._T(x), new_prms)

    def value_shape(self):
        return ()


def dc_imp(T, prms):
    flux = prms["implanted_flux"]
    implantation_depth = prms["implantation_depth"]
    D = prms["D_0"]*exp(-prms["E_D"]/FESTIM.k_B/T)
    value = flux*implantation_depth/D
    if "Kr_0" in prms:
        Kr = prms["Kr_0"]*exp(-prms["E_Kr"]/FESTIM.k_B/T)
        value += (flux/Kr)**0.5

    return value


def sieverts_law(T, prms):
    S_0, E_S = prms["S_0"], prms["E_S"]
    S = S_0*exp(-E_S/FESTIM.k_B/T)
    return S*prms["pressure"]**0.5


def apply_boundary_conditions(simulation):
    """Create a list of DirichletBCs.

    Arguments:


    Raises:
        KeyError: Raised if the type key of bc is missing
        NameError: Raised if type is unknown

    Returns:
        list -- contains fenics DirichletBC
        list -- contains the fenics.Expression() to be updated
    """
    parameters = simulation.parameters
    V = simulation.V
    boundary_conditions = parameters["boundary_conditions"]
    volume_markers = simulation.volume_markers
    surface_markers = simulation.surface_markers
    T = simulation.T

    bcs = list()
    expressions = list()

    for BC in boundary_conditions:
        if "type" in BC.keys():
            type_BC = BC["type"]
        else:
            raise KeyError("Missing boundary condition type key")
        if type_BC == "dc":
            value_BC = sp.printing.ccode(BC['value'])
            value_BC = Expression(value_BC, t=0, degree=4)
        elif type_BC == "solubility":
            prms = {
                "pressure": BC["pressure"],
                "S_0": BC["S_0"],
                "E_S": BC["E_S"],
            }

            # create a custom expression
            value_BC = BoundaryCondition(T, prms, eval_function=sieverts_law)
            expressions.append(value_BC.prms["pressure"])
            expressions.append(value_BC.prms["S_0"])
            expressions.append(value_BC.prms["E_S"])
        elif type_BC == "dc_imp":
            prms = {
                "implanted_flux": BC["implanted_flux"],
                "implantation_depth": BC["implantation_depth"],
                "D_0": BC["D_0"],
                "E_D": BC["E_D"],
            }
            if "Kr_0" in BC.keys() and "E_Kr" in BC.keys():
                prms["Kr_0"] = BC["Kr_0"]
                prms["E_Kr"] = BC["E_Kr"]

            value_BC = BoundaryCondition(T, prms, eval_function=dc_imp)
            expressions.append(value_BC.prms["implanted_flux"])
            expressions.append(value_BC.prms["implantation_depth"])

        if BC["type"] not in FESTIM.helpers.bc_types["neumann"] and \
           BC["type"] not in FESTIM.helpers.bc_types["robin"] and \
           BC["type"] not in FESTIM.helpers.bc_types["dc"]:

            raise NameError("Unknown boundary condition type : " + BC["type"])
        if type_BC in FESTIM.helpers.bc_types["dc"]:
            if "component" in BC.keys():
                # Fetch the component of the BC
                component = BC["component"]
            else:
                # By default, component is solute (ie. 0)
                component = 0
            conservation_chemic_pot = False
            for mat in parameters["materials"]:
                if "S_0" in mat.keys():
                    conservation_chemic_pot = True
            if component == 0 and conservation_chemic_pot is True:
                # Store the non modified BC to be updated
                expressions.append(value_BC)
                # create modified BC based on solubility
                value_BC = BoundaryConditionTheta(
                    value_BC, volume_markers.mesh(), parameters["materials"],
                    volume_markers, T)
            expressions.append(value_BC)

            surfaces = BC['surfaces']
            if type(surfaces) is not list:
                surfaces = [surfaces]
            if V.num_sub_spaces() == 0:
                funspace = V
            else:  # if only one component, use subspace
                funspace = V.sub(component)
            for surface in surfaces:
                bci = DirichletBC(funspace, value_BC,
                                  surface_markers, surface)
                bcs.append(bci)

    return bcs, expressions
