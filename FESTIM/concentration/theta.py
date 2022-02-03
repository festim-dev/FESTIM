from FESTIM import Mobile, k_B
import fenics as f


class Theta(Mobile):
    """Class representing the "chemical potential" c/S where S is the
    solubility of the metal
    """
    def __init__(self):
        """Inits Theta
        """
        super().__init__()
        self.S = None
        self.materials = None
        self.volume_markers = None
        self.T = None

    def initialise(self, V, value, label=None, time_step=None):
        """Assign a value to self.previous_solution

        Args:
            V (fenics.FunctionSpace): the function space
            value (sp.Add, float, int, str): the value of the initialisation.
            label (str, optional): the label in the XDMF file. Defaults to
                None.
            time_step (int, optional): the time step to read in the XDMF file.
                Defaults to None.
        """
        comp = self.get_comp(V, value, label=label, time_step=time_step)
        comp = ConcentrationToTheta(
            comp, self.materials, self.volume_markers, self.T.T
        )

        # Product must be projected
        comp = f.project(comp, V)
        f.assign(self.previous_solution, comp)

    def get_concentration_for_a_given_material(self, material, T):
        """Returns the concentration (and previous concentration) for a given
        material

        Args:
            material (FESTIM.Material): the material with attributes S_0 and
                E_S
            T (FESTIM.Temperature): the temperature with attributest T and T_n

        Returns:
            fenics.Product, fenics.Product: the current concentration and
                previous concentration
        """
        # TODO this needs changing for Henry
        E_S = material.E_S
        S_0 = material.S_0
        c_0 = self.solution*S_0*f.exp(-E_S/k_B/T.T)
        c_0_n = self.previous_solution*S_0*f.exp(-E_S/k_B/T.T_n)
        return c_0, c_0_n

    def mobile_concentration(self):
        """Returns the hydrogen concentration as c=theta*S
        Where S is FESTIM.ArheniusCoeff defines on all materials.
        This is needed when adding neuman or robin BCs to the form.

        Returns:
            fenics.Product: the hydrogen mobile concentration
        """
        # TODO this needs changing for Henry
        # return self.solution*self.S
        return ThetaToConcentration(
            self.solution, self.materials, self.volume_markers, self.T.T
        )

    def post_processing_solution_to_concentration(self):
        """Converts the post_processing_solution from theta to mobile
        concentration.
        c = theta * S.
        The attribute post_processing_solution is fenics.Product (if self.S is
        FESTIM.ArheniusCoeff)
        """
        # TODO this needs changing for Henry
        self.post_processing_solution *= self.S
        # extremely slow
        # self.post_processing_solution = ThetaToConcentration(
        #     self.post_processing_solution, self.materials, self.volume_markers, self.S
        # )


# TODO merge this with dirichlet_bc.BoundaryConditionTheta
class ConcentrationToTheta(f.UserExpression):
    """Creates an Expression for converting dirichlet bcs in the case
    of chemical potential conservation
    """
    def __init__(self, comp, materials, vm, T, **kwargs):
        """initialisation

        Args:
            comp (fenics.Expression): value of BC
            materials (FESTIM.Materials): contains materials objects
            vm (fenics.MeshFunction): volume markers
            T (fenics.Function): Temperature
        """
        super().__init__(kwargs)
        self._comp = comp
        self._vm = vm
        self._T = T
        self._materials = materials

    def eval_cell(self, value, x, ufc_cell):
        cell = f.Cell(self._vm.mesh(), ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = self._materials.find_material_from_id(subdomain_id)
        # TODO this requires changes for Henry's law
        if material.solubility_law == "sieverts":
            S_0 = material.S_0
            E_S = material.E_S
            c = self._comp(x)
            S = S_0*f.exp(-E_S/k_B/self._T(x))
            value[0] = c/S
        else:
            assert False

    def value_shape(self):
        return ()


class ThetaToConcentration(f.UserExpression):
    def __init__(self, theta, materials, vm, S, **kwargs):
        """initialisation

        Args:
            theta (fenics.Expression): value of BC
            materials (FESTIM.Materials): contains materials objects
            vm (fenics.MeshFunction): volume markers
            T (fenics.Function): Temperature
        """
        super().__init__(kwargs)
        self._theta = theta
        self._vm = vm
        self._materials = materials
        self._S = S

    def eval_cell(self, value, x, ufc_cell):
        cell = f.Cell(self._vm.mesh(), ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = self._materials.find_material_from_id(subdomain_id)
        # TODO this requires changes for Henry's law
        if material.solubility_law == "sieverts":
            theta = self._theta(x)
            S = self._S(x)
            value[0] = theta*S
        else:
            assert False

    def value_shape(self):
        return ()
