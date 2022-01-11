import FESTIM
import fenics as f
import sympy as sp


class BoundaryCondition:
    def __init__(self, type, surfaces, value=None, function=None, component=0, **kwargs) -> None:
        self.type = type
        self.check_type()

        if not isinstance(surfaces, list):
            surfaces = [surfaces]
        self.surfaces = surfaces

        self.value = value
        self.function = function
        self.component = component
        self.prms = kwargs
        self.expression = None
        self.sub_expressions = []

    def check_type(self):
        print(self.type)
        if self.type not in FESTIM.helpers.bc_types["neumann"] and \
           self.type not in FESTIM.helpers.bc_types["robin"] and \
           self.type not in FESTIM.helpers.bc_types["dc"]:
            raise NameError(
                    "Unknown boundary condition type : " + self.type)


class DirichletBC(BoundaryCondition):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def create_expression(self, T):
        if self.type == "dc":
            value_BC = sp.printing.ccode(self.value)
            value_BC = f.Expression(value_BC, t=0, degree=4)
            # TODO : why degree 4?
        else:
            if self.type == "dc_custom":
                function = self.function
            else:
                function = type_to_function[self.type]

            ignored_keys = ["type", "surfaces", "function", "component"]

            prms = {key: val for key, val in self.prms.items() if key not in ignored_keys}
            value_BC = BoundaryConditionExpression(T, prms, eval_function=function)
            self.sub_expressions = [value_BC.prms[key] for key in prms.keys()]

        self.expression = value_BC
        return value_BC

    def normalise_by_solubility(self, simulation):
        # Store the non modified BC to be updated
        self.sub_expressions.append(self.expression)
        # create modified BC based on solubility
        expression_BC = BoundaryConditionTheta(
                            self.expression,
                            simulation.parameters["materials"],
                            simulation.volume_markers, simulation.T)
        self.expression = expression_BC
        return expression_BC


class FluxBC(BoundaryCondition):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def create_form_for_flux(self, T, solute):
        if self.type == "flux":
            form = sp.printing.ccode(self.value)
            form = f.Expression(form, t=0, degree=2)
            self.sub_expressions.append(form)
        elif self.type == "recomb":
            form = recombination_flux(T, solute, self.prms)
        elif self.type == "convective_flux":
            form = convective_flux(T, self.prms)
        elif self.type == "flux_custom":
            prms = {}
            for key, val in self.prms.items():
                if isinstance(val, (int, float)):
                    prms[key] = f.Constant(val)
                else:
                    prms[key] = f.Expression(sp.printing.ccode(val), t=0, degree=1)
            form = self.function(T, solute, prms)
            self.sub_expressions += [expression for expression in prms.values()]
        self.form = form
        return form


def define_dirichlet_bcs_T(simulation):
    """Creates a list of BCs for thermal problem

    Arguments:


    Returns:
        list -- contains fenics.DirichletBC
        list -- contains fenics.Expression to be updated
    """

    bcs = []
    expressions = []
    for bc in simulation.boundary_conditions:
        if bc.component == "T" and bc.type == "dc":
            bc.create_expression(simulation.T)
            # TODO: one day, we will get rid of this big expressions list
            expressions += bc.sub_expressions
            expressions.append(bc.expression)

            for surf in bc.surfaces:
                bci = f.DirichletBC(
                    simulation.V_CG1, bc.expression,
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
    solutions = f.split(simulation.u)
    test_solute = f.split(simulation.v)[0]
    F = 0

    if simulation.chemical_pot:
        solute = solutions[0]*simulation.S
    else:
        solute = solutions[0]

    for bc in simulation.boundary_conditions:
        if bc.component != "T":
            if bc.type not in FESTIM.helpers.bc_types["dc"]:
                bc.create_form_for_flux(simulation.T, solute)
                # TODO : one day we will get rid of this huge expressions list
                expressions += bc.sub_expressions

                for surf in bc.surfaces:
                    F += -test_solute*bc.form*simulation.ds(surf)
    return F, expressions


def apply_heat_fluxes(simulation):
    F = 0
    expressions = []
    vT = simulation.vT
    ds = simulation.ds
    T = simulation.T
    for bc in simulation.boundary_conditions:
        if bc.component == "T":
            if bc.type not in FESTIM.helpers.T_bc_types["dc"]:
                bc.create_form_for_flux(T, solute=None)
                expressions += bc.sub_expressions

                for surf in bc.surfaces:
                    F += -bc.form*vT*ds(surf)
    return F, expressions


class BoundaryConditionTheta(f.UserExpression):
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
        cell = f.Cell(self._mesh, ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = FESTIM.helpers.find_material_from_id(
            self._materials, subdomain_id)
        S_0 = material["S_0"]
        E_S = material["E_S"]
        value[0] = self._bci(x)/(S_0*f.exp(-E_S/FESTIM.k_B/self._T(x)))

    def value_shape(self):
        return ()


class BoundaryConditionExpression(f.UserExpression):
    def __init__(self, T, prms, eval_function):

        super().__init__()
        # create Expressions or Constant for all parameters
        for key, value in prms.items():
            if isinstance(value, (int, float)):
                prms[key] = f.Constant(value)
            else:
                prms[key] = f.Expression(sp.printing.ccode(value),
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


def recombination_flux(T, c, prms):
    Kr = prms["Kr_0"]*f.exp(-prms["E_Kr"]/FESTIM.k_B/T)
    return -Kr*c**prms["order"]


def dc_imp(T, prms):
    flux = prms["implanted_flux"]
    implantation_depth = prms["implantation_depth"]
    D = prms["D_0"]*f.exp(-prms["E_D"]/FESTIM.k_B/T)
    value = flux*implantation_depth/D
    if "Kr_0" in prms:
        Kr = prms["Kr_0"]*f.exp(-prms["E_Kr"]/FESTIM.k_B/T)
        value += (flux/Kr)**0.5

    return value


def sieverts_law(T, prms):
    S_0, E_S = prms["S_0"], prms["E_S"]
    S = S_0*f.exp(-E_S/FESTIM.k_B/T)
    return S*prms["pressure"]**0.5


def convective_flux(T, prms):
    return prms["h_coeff"] * (T - prms["T_ext"])


type_to_function = {
    "solubility": sieverts_law,
    "dc_imp": dc_imp,
}


def define_dirichlet_bcs(simulation):
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

    #  for BC_object in simulation.boundary_conditions:
    for BC_object in simulation.boundary_conditions:
        if BC_object.component != "T":
            BC_object.create_expression(simulation.T)

            if BC_object.type in FESTIM.helpers.bc_types["dc"]:

                if BC_object.component == 0 and simulation.chemical_pot:
                    BC_object.normalise_by_solubility(simulation)

                # TODO: one day, we will get rid of this big expressions list
                expressions += BC_object.sub_expressions
                # add value_BC to expressions for update
                expressions.append(BC_object.expression)

                # create a DirichletBC and add it to bcs
                if simulation.V.num_sub_spaces() == 0:
                    funspace = simulation.V
                else:  # if only one component, use subspace
                    funspace = simulation.V.sub(BC_object.component)
                for surface in BC_object.surfaces:
                    bci = f.DirichletBC(funspace, BC_object.expression,
                                    simulation.surface_markers, surface)
                    bcs.append(bci)

    return bcs, expressions
