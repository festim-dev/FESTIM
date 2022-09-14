from festim import BoundaryCondition, k_B
import fenics as f
import sympy as sp


class DirichletBC(BoundaryCondition):
    """Class to enforce the solution on boundaries.

    Args:
        surfaces (list or int): the surfaces of the BC
        value (float or sp.Expr): the value of the boundary
            condition.
        field (int or str): the field the boundary condition is
            applied to. 0 and "solute" stand for the mobile
            concentration, "T" for temperature
    """

    def __init__(self, surfaces, value, field) -> None:
        super().__init__(surfaces, field=field)
        self.value = value
        self.dirichlet_bc = []

    def create_expression(self, T):
        """Assigns a value to self.expression

        Args:
            T (fenics.Function): temperature
        """
        value_BC = sp.printing.ccode(self.value)
        value_BC = f.Expression(value_BC, t=0, degree=4)
        # TODO : why degree 4?

        self.expression = value_BC

    def normalise_by_solubility(self, materials, volume_markers, T):
        """Normalise self.expression by the solubility
        theta = c/S

        Args:
            materials (festim.Materials): the materials
            volume_markers (fenics.MeshFunction): the volume markers
            T (fenics.Function): the temperature
        """
        # TODO this requires changes for Henry's law

        # Store the non modified BC to be updated
        self.sub_expressions.append(self.expression)
        # create modified BC based on solubility
        expression_BC = BoundaryConditionTheta(
            self.expression, materials, volume_markers, T
        )
        self.expression = expression_BC

    def create_dirichletbc(
        self,
        V,
        T,
        surface_markers,
        chemical_pot=False,
        materials=None,
        volume_markers=None,
    ):
        """creates a list of fenics.DirichletBC and stores it in
        self.dirichlet_bc

        Args:
            V (fenics.FunctionSpace): the function space of the field
            T (fenics.Constant or fenics.Expression or fenics.Function): the
                temperature
            surface_markers (fenics.MeshFunction): the surface markers
            chemical_pot (bool, optional): if True, conservation of chemical
                pot will be assumed. Defaults to False.
            materials (festim.Materials): The materials, only needed when
                chemical_pot is True. Defaults to None.
            volume_markers (fenics.MeshFunction, optional): the volume markers,
                only needed when chemical_pot is True. Defaults to None.
        """
        self.dirichlet_bc = []
        self.create_expression(T)
        # TODO: this should be more generic
        mobile_fields = [0, "0", "solute"]
        if self.field in mobile_fields and chemical_pot:
            self.normalise_by_solubility(materials, volume_markers, T)

        # create a DirichletBC and add it to bcs
        if V.num_sub_spaces() == 0:
            funspace = V
        else:  # if only one field, use subspace
            funspace = V.sub(self.field)
        for surface in self.surfaces:
            bci = f.DirichletBC(funspace, self.expression, surface_markers, surface)
            self.dirichlet_bc.append(bci)


class BoundaryConditionTheta(f.UserExpression):
    """Creates an Expression for converting dirichlet bcs in the case
    of chemical potential conservation

    Args:
        bci (fenics.Expression): value of BC
        mesh (fenics.mesh): mesh
        materials (festim.Materials): contains materials objects
        vm (fenics.MeshFunction): volume markers
        T (fenics.Function): Temperature
    """

    def __init__(self, bci, materials, vm, T, **kwargs):
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
        c = self._bci(x)
        S = S_0 * f.exp(-E_S / k_B / self._T(x))
        if material.solubility_law == "sievert":
            value[0] = c / S
        elif material.solubility_law == "henry":
            value[0] = (c / S + f.DOLFIN_EPS) ** 0.5

    def value_shape(self):
        return ()


class BoundaryConditionExpression(f.UserExpression):
    """ "[summary]"

    Args:
        T (fenics.Function): the temperature
        eval_function ([type]): [description]
    """

    def __init__(self, T, eval_function, **kwargs):

        super().__init__()

        self._T = T
        self.eval_function = eval_function
        self.prms = kwargs

    def eval(self, value, x):
        # find local value of parameters
        new_prms = {}
        for key, prm_val in self.prms.items():
            if callable(prm_val):
                if isinstance(prm_val, f.Constant):
                    new_prms[key] = float(prm_val)
                else:
                    new_prms[key] = prm_val(x)
            else:
                new_prms[key] = prm_val

        # evaluate at local point
        value[0] = self.eval_function(self._T(x), **new_prms)

    def value_shape(self):
        return ()
