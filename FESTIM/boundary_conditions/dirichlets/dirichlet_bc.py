import FESTIM
import fenics as f
import sympy as sp


class DirichletBC(FESTIM.BoundaryCondition):
    def __init__(self, surfaces, value=None, component=0) -> None:
        super().__init__(surfaces, component=component)
        self.value = value
        self.dirichlet_bc = []

    def create_expression(self, T):
        """[summary]

        Args:
            T (fenics.Function): temperature

        Returns:
            [type]: [description]
        """
        value_BC = sp.printing.ccode(self.value)
        value_BC = f.Expression(value_BC, t=0, degree=4)
        # TODO : why degree 4?

        self.expression = value_BC

    def normalise_by_solubility(self, materials, volume_markers, T):
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
    def __init__(self, T, eval_function, **kwargs):
        """"[summary]"

        Args:
            T (fenics.Function): the temperature
            eval_function ([type]): [description]
        """

        super().__init__()

        self._T = T
        self.eval_function = eval_function
        self.prms = kwargs

    def eval(self, value, x):
        # find local value of parameters
        new_prms = {}
        for key, prm_val in self.prms.items():
            if callable(prm_val):
                new_prms[key] = prm_val(x)
            else:
                new_prms[key] = prm_val

        # evaluate at local point
        value[0] = self.eval_function(self._T(x), **new_prms)

    def value_shape(self):
        return ()
