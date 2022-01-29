from operator import itemgetter
import warnings
import numpy as np
from FESTIM import k_B
import fenics as f


class Material:
    def __init__(self, id, D_0, E_D, S_0=None, E_S=None, thermal_cond=None, heat_capacity=None, rho=None, borders=[], H=None) -> None:
        """Inits Material class

        Args:
            id (int): the id of the material
            D_0 (float): diffusion coefficient pre-exponential factor (m2/s)
            E_D (float): diffusion coefficient activation energy (eV)
            S_0 (float, optional): Solubility pre-exponential factor
                (H/m3/Pa0.5). Defaults to None.
            E_S (float, optional): Solubility activation energy (eV).
                Defaults to None.
            thermal_cond (float or callable, optional): thermal conductivity
                (W/m/K). Can be a function of T. Defaults to None.
            heat_capacity (float or callable, optional): heat capacity
                (J/K/kg). Can be a function of T. Defaults to None.
            rho (float or callable, optional): volumetric density (kg/m3). Can
                be a function of T. Defaults to None.
            borders (list, optional): The borders of the 1D subdomain.
                Only needed in 1D with several materials. Defaults to [].
            H (dict, optional): heat of transport (J/mol).
                {"free_enthalpy": ..., "entropy": ...} so that
                H = free_enthalpy + entropy*T. Defaults to None.
        """
        self.id = id
        self.D_0 = D_0
        self.E_D = E_D
        self.S_0 = S_0
        self.E_S = E_S
        self.thermal_cond = thermal_cond
        self.heat_capacity = heat_capacity
        self.rho = rho
        self.borders = borders
        self.H = H
        if H is not None:
            self.free_enthalpy = H["free_enthalpy"]
            self.entropy = H["entropy"]
        self.check_properties()

    def check_properties(self):
        """Checks that if S_0 is None E_S is not None and reverse.

        Raises:
            ValueError: [description]
            ValueError: [description]
        """
        if self.S_0 is None and self.E_S is not None:
            raise ValueError("S_0 cannot be None")
        if self.E_S is None and self.S_0 is not None:
            raise ValueError("E_S cannot be None")


class Materials:
    def __init__(self, materials=[]):
        """Inits Materials

        Args:
            materials (list, optional): contains FESTIM.Material objects.
                Defaults to [].
        """
        self.materials = materials
        self.D = None
        self.S = None
        self.thermal_cond = None
        self.heat_capacity = None
        self.density = None
        self.H = None

    def check_borders(self, size):
        """Checks that the borders of the materials match

        Args:
            size (float): size of the 1D domain

        Raises:
            ValueError: if the borders don't begin at zero
            ValueError: if borders don't match
            ValueError: if borders don't end at size

        Returns:
            bool -- True if everything's alright
        """
        all_borders = []
        for m in self.materials:
            all_borders.append(m.borders)
        all_borders = sorted(all_borders, key=itemgetter(0))
        if all_borders[0][0] is not 0:
            raise ValueError("Borders don't begin at zero")
        for i in range(0, len(all_borders)-1):
            if all_borders[i][1] != all_borders[i+1][0]:
                raise ValueError("Borders don't match to each other")
        if all_borders[len(all_borders) - 1][1] != size:
            raise ValueError("Borders don't match with size")
        return True

    def check_materials(self, temp_type, derived_quantities={}):
        """Checks the materials keys

        Args:
            temp_type (str): the type of FESTIM.Temperature
            derived_quantities (dict, optional): [description]. Defaults to {}.
        """

        if len(self.materials) > 0:  # TODO: get rid of this...
            self.check_consistency()

            self.check_for_unused_properties(temp_type, derived_quantities)

            self.check_unique_ids()

            self.check_missing_properties(temp_type, derived_quantities)

    def check_unique_ids(self):
        # check that ids are different
        mat_ids = []
        for mat in self.materials:
            if type(mat.id) is list:
                mat_ids += mat.id
            else:
                mat_ids.append(mat.id)

        if len(mat_ids) != len(np.unique(mat_ids)):
            raise ValueError("Some materials have the same id")

    def check_for_unused_properties(self, temp_type, derived_quantities):
        # warn about unused keys
        transient_properties = ["rho", "heat_capacity"]
        if temp_type != "solve_transient":
            for mat in self.materials:
                for key in transient_properties:
                    if getattr(mat, key) is not None:
                        warnings.warn(key + " key will be ignored", UserWarning)

        for mat in self.materials:
            if getattr(mat, "thermal_cond") is not None:
                warn = True
                if temp_type != "expression":
                    warn = False
                elif "surface_flux" in derived_quantities:
                    for surface_flux in derived_quantities["surface_flux"]:
                        if surface_flux["field"] == "T":
                            warn = False
                if warn:
                    warnings.warn("thermal_cond key will be ignored", UserWarning)

    def check_consistency(self):
        # check the materials keys match
        attributes = {
            "S_0": [],
            "E_S": [],
            "thermal_cond": [],
            "heat_capacity": [],
            "rho": [],
            "borders": [],
            "H": [],
        }

        for attr, value in attributes.items():
            for mat in self.materials:
                value.append(getattr(mat, attr))
            if value.count(None) not in [0, len(self.materials)]:
                raise ValueError("{} is not defined for all materials".format(attr))

    def check_missing_properties(self, temp_type, derived_quantities):
        if temp_type != "expression" and \
                self.materials[0].thermal_cond is None:
            raise NameError("Missing thermal_cond key in materials")
        if temp_type == "solve_transient":
            if self.materials[0].heat_capacity is None:
                raise NameError("Missing heat_capacity key in materials")
            if self.materials[0].rho is None:
                raise NameError("Missing rho key in materials")
        # TODO: add check for thermal cond for thermal flux computation

    def find_material_from_id(self, mat_id):
        """Returns the material from a given id

        Args:
            mat_id (int): id of the wanted material

        Raises:
            ValueError: if the id isn't found

        Returns:
            FESTIM.Material: the material that has the id mat_id
        """
        for material in self.materials:
            mat_ids = material.id
            if type(mat_ids) is not list:
                mat_ids = [mat_ids]
            if mat_id in mat_ids:
                return material
        raise ValueError("Couldn't find ID " + str(mat_id) + " in materials list")

    def create_properties(self, vm, T):
        """Creates the properties fields needed for post processing

        Arguments:
            vm {fenics.MeshFunction()} -- volume markers
            T {fenics.Function()} -- temperature
        """
        self.D = ArheniusCoeff(self, vm, T, "D_0", "E_D", degree=2)
        # all materials have the same properties so only checking the first is enough
        if self.materials[0].S_0 is not None:
            self.S = ArheniusCoeff(self, vm, T, "S_0", "E_S", degree=2)
        if self.materials[0].thermal_cond is not None:
            self.thermal_cond = ThermalProp(self, vm, T,
                                        'thermal_cond', degree=2)
            self.heat_capacity = ThermalProp(self, vm, T,
                                'heat_capacity', degree=2)
            self.density = ThermalProp(self, vm, T,
                                'rho', degree=2)
        if self.materials[0].H is not None:
            self.H = HCoeff(self, vm, T, degree=2)


class ArheniusCoeff(f.UserExpression):
    def __init__(self, materials, vm, T, pre_exp, E, **kwargs):
        super().__init__(kwargs)
        self._vm = vm
        self._T = T
        self._materials = materials
        self._pre_exp = pre_exp
        self._E = E

    def eval_cell(self, value, x, ufc_cell):
        cell = f.Cell(self._vm.mesh(), ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = self._materials.find_material_from_id(subdomain_id)
        D_0 = getattr(material, self._pre_exp)
        E_D = getattr(material, self._E)
        value[0] = D_0*f.exp(-E_D/k_B/self._T(x))

    def value_shape(self):
        return ()


class ThermalProp(f.UserExpression):
    def __init__(self, materials, vm, T, key, **kwargs):
        super().__init__(kwargs)
        self._T = T
        self._vm = vm
        self._materials = materials
        self._key = key

    def eval_cell(self, value, x, ufc_cell):
        cell = f.Cell(self._vm.mesh(), ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = self._materials.find_material_from_id(subdomain_id)
        attribute = getattr(material, self._key)
        if callable(attribute):
            value[0] = attribute(self._T(x))
        else:
            value[0] = attribute

    def value_shape(self):
        return ()


class HCoeff(f.UserExpression):
    def __init__(self, materials, vm, T, **kwargs):
        super().__init__(kwargs)
        self._T = T
        self._vm = vm
        self._materials = materials

    def eval_cell(self, value, x, ufc_cell):
        cell = f.Cell(self._vm.mesh(), ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = self._materials.find_material_from_id(subdomain_id)

        value[0] = material.free_enthalpy + \
            self._T(x)*material.entropy

    def value_shape(self):
        return ()

