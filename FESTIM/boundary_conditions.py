import FESTIM
from fenics import *
import sympy as sp


def define_dirichlet_bcs_T(simulation):
    """Creates a list of BCs for thermal problem

    Arguments:


    Returns:
        list -- contains fenics.DirichletBC
        list -- contains fenics.Expression to be updated
    """

    bcs = []
    expressions = []
    for bc in simulation.parameters["temperature"]["boundary_conditions"]:
        if bc["type"] == "dc":
            expression_bc = sp.printing.ccode(bc["value"])
            expression_bc = Expression(expression_bc, degree=2, t=0)
            expressions.append(expression_bc)
            if type(bc["surfaces"]) is list:
                surfaces = bc["surfaces"]
            else:
                surfaces = [bc["surfaces"]]
            for surf in surfaces:
                bci = DirichletBC(
                    simulation.V_CG1, expression_bc,
                    simulation.surface_markers, surf)
                bcs.append(bci)
    return bcs, expressions


def apply_fluxes(simulation):
    """Modifies the formulation and adds fluxes based
    on parameters in boundary_conditions

    Arguments:

    Raises:
        NameError: if boundary condition type is unknown

    Returns:
        fenics.Form() -- formulation for BCs
        list -- contains all the fenics.Expression() to be updated
    """

    expressions = []
    solutions = split(simulation.u)
    test_solute = split(simulation.v)[0]
    F = 0

    if simulation.chemical_pot:
        solute = solutions[0]*simulation.S
    else:
        solute = solutions[0]

    for bc in simulation.parameters["boundary_conditions"]:
        if bc["type"] not in FESTIM.helpers.bc_types["dc"]:
            check_flux_bc_type(bc)
            flux = create_form_for_flux(simulation.T, expressions, solute, bc)

            if type(bc['surfaces']) is not list:
                surfaces = [bc['surfaces']]
            else:
                surfaces = bc['surfaces']
            for surf in surfaces:
                F += -test_solute*flux*simulation.ds(surf)
    return F, expressions


def create_form_for_flux(T, expressions, solute, bc):
    if bc["type"] == "flux":
        flux = sp.printing.ccode(bc["value"])
        flux = Expression(flux, t=0, degree=2)
        expressions.append(flux)
    elif bc["type"] == "recomb":
        Kr = bc["Kr_0"]*exp(-bc["E_Kr"]/FESTIM.k_B/T)
        flux = -Kr*solute**bc["order"]
    elif bc["type"] == "flux_custom":
        ignored_keys = ["type", "surfaces", "function"]
        prms = {}
        for key, val in bc.items():
            if key not in ignored_keys:
                prms[key] = Expression(sp.printing.ccode(val), t=0, degree=1)
        flux = bc["function"](T, solute, prms)
        expressions += [expression for expression in prms.values()]
    return flux


def check_flux_bc_type(bc):
    if bc["type"] not in FESTIM.helpers.bc_types["neumann"] and \
               bc["type"] not in FESTIM.helpers.bc_types["robin"]:
        raise NameError(
                    "Unknown boundary condition type : " + bc["type"])


class BoundaryConditionTheta(UserExpression):
    """Creates an Expression for converting dirichlet bcs in the case
    of chemical potential conservation

    Args:
        UserExpression (fenics.UserExpression):
    """
    def __init__(self, bci, materials, vm, T, **kwargs):
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
        self._vm = vm
        self._mesh = vm.mesh()
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


class BoundaryConditionExpression(UserExpression):
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


type_to_function = {
    "solubility": sieverts_law,
    "dc_imp": dc_imp,
}


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

    bcs = list()
    expressions = list()

    for BC in simulation.parameters["boundary_conditions"]:
        check_type(BC)
        expression_BC = create_bc_expression(BC, simulation.T, expressions)

        if BC["type"] in FESTIM.helpers.bc_types["dc"]:
            # By default, component is solute (ie. 0)
            component = 0
            if "component" in BC.keys():
                # Fetch the component of the BC
                component = BC["component"]

            if component == 0 and simulation.chemical_pot:
                expression_BC = normalise_expression_by_S(
                    simulation, expressions, expression_BC)

            # add value_BC to expressions for update
            expressions.append(expression_BC)

            # create a DirichletBC and add it to bcs
            surfaces = BC['surfaces']
            if type(surfaces) is not list:
                surfaces = [surfaces]
            if simulation.V.num_sub_spaces() == 0:
                funspace = simulation.V
            else:  # if only one component, use subspace
                funspace = simulation.V.sub(component)
            for surface in surfaces:
                bci = DirichletBC(funspace, expression_BC,
                                  simulation.surface_markers, surface)
                bcs.append(bci)

    return bcs, expressions


def normalise_expression_by_S(simulation, expressions, expression_BC):
    # Store the non modified BC to be updated
    expressions.append(expression_BC)
    # create modified BC based on solubility
    expression_BC = BoundaryConditionTheta(
                    expression_BC,
                    simulation.parameters["materials"],
                    simulation.volume_markers, simulation.T)

    return expression_BC


def check_type(BC):
    if "type" in BC.keys():
        if BC["type"] not in FESTIM.helpers.bc_types["neumann"] and \
               BC["type"] not in FESTIM.helpers.bc_types["robin"] and \
               BC["type"] not in FESTIM.helpers.bc_types["dc"]:
            raise NameError(
                    "Unknown boundary condition type : " + BC["type"])
    else:
        raise KeyError("Missing boundary condition type key")


def create_bc_expression(BC, T, expressions):

    if BC["type"] == "dc":
        value_BC = sp.printing.ccode(BC['value'])
        value_BC = Expression(value_BC, t=0, degree=4)
    else:
        if BC["type"] == "dc_custom":
            function = BC["function"]
        else:
            function = type_to_function[BC["type"]]

        ignored_keys = ["type", "surfaces", "function", "component"]

        prms = {key: val for key, val in BC.items() if key not in ignored_keys}
        value_BC = BoundaryConditionExpression(T, prms, eval_function=function)
        expressions += [value_BC.prms[key] for key in prms.keys()]
    return value_BC
