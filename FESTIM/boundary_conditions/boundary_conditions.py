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
        self.convert_prms()

    def convert_prms(self):
        # create Expressions or Constant for all parameters
        for key, value in self.prms.items():
            if isinstance(value, (int, float)):
                self.prms[key] = f.Constant(value)
            else:
                self.prms[key] = f.Expression(sp.printing.ccode(value),
                                       t=0,
                                       degree=1)

    def check_type(self):
        possible_types = FESTIM.helpers.bc_types["neumann"] + \
            FESTIM.helpers.bc_types["robin"] + \
            FESTIM.helpers.bc_types["dc"]
        possible_types += FESTIM.helpers.T_bc_types["neumann"] + \
            FESTIM.helpers.T_bc_types["robin"] + \
            FESTIM.helpers.T_bc_types["dc"]
        if self.type not in possible_types:
            raise NameError(
                    "Unknown boundary condition type : " + self.type)


class DirichletBC(BoundaryCondition):
    def __init__(self, type, surfaces, value=None, function=None, component=0, **kwargs) -> None:
        super().__init__(type, surfaces, value=value, function=function, component=component, **kwargs)
        self.dirichlet_bc = []

    def create_expression(self, T):
        """[summary]

        Args:
            T (fenics.Function): temperature

        Returns:
            [type]: [description]
        """
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

    def normalise_by_solubility(self, materials, volume_markers, T):
        print('coucou')
        # Store the non modified BC to be updated
        self.sub_expressions.append(self.expression)
        # create modified BC based on solubility
        expression_BC = BoundaryConditionTheta(
                            self.expression,
                            materials,
                            volume_markers, T)
        self.expression = expression_BC
        return expression_BC

    def create_dirichletbc(self, V, T, surface_markers, chemical_pot=False, materials=None, volume_markers=None):
        """[summary]

        Args:
            V (fenics.FunctionSpace): [description]
            T (fenics.Constant or fenics.Expression or fenics.Function): [description]
            surface_markers (fenics.MeshFunction): [description]
            chemical_pot (bool, optional): [description]. Defaults to False.
            materials ([type], optional): [description]. Defaults to None.
            volume_markers ([type], optional): [description]. Defaults to None.
        """
        self.dirichlet_bc = []
        self.create_expression(T)
        # TODO: this should be more generic
        mobile_components = [0, "0", "solute"]
        if self.component in mobile_components and chemical_pot:
            self.normalise_by_solubility(materials, volume_markers, T)

        # create a DirichletBC and add it to bcs
        if V.num_sub_spaces() == 0:
            funspace = V
        else:  # if only one component, use subspace
            funspace = V.sub(self.component)
        for surface in self.surfaces:
            bci = f.DirichletBC(funspace, self.expression,
                                surface_markers, surface)
            self.dirichlet_bc.append(bci)


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
            materials (FESTIM.Materials): contains materials objects
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
        material = self._materials.find_material_from_id(subdomain_id)
        S_0 = material.S_0
        E_S = material.E_S
        value[0] = self._bci(x)/(S_0*f.exp(-E_S/FESTIM.k_B/self._T(x)))

    def value_shape(self):
        return ()


class BoundaryConditionExpression(f.UserExpression):
    def __init__(self, T, prms, eval_function):
        """"[summary]"

        Args:
            T (fenics.Function): the temperature
            prms (dict): [description]
            eval_function ([type]): [description]
        """

        super().__init__()

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


type_to_function = {
    "solubility": sieverts_law,
    "dc_imp": dc_imp,
}
