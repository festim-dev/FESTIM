from operator import itemgetter
import warnings
import numpy as np
from festim import k_B, Material, HeatTransferProblem
import festim
import fenics as f
from typing import Union


class Materials:
    """
    Args:
        materials (list, optional): contains festim.Material objects.
            Defaults to [].
    """

    def __init__(self, materials=[]):

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
            if isinstance(m.borders[0], list):
                for border in m.borders:
                    all_borders.append(border)
            else:
                all_borders.append(m.borders)
        all_borders = sorted(all_borders, key=itemgetter(0))
        if all_borders[0][0] is not 0:
            raise ValueError("Borders don't begin at zero")
        for i in range(0, len(all_borders) - 1):
            if all_borders[i][1] != all_borders[i + 1][0]:
                raise ValueError("Borders don't match to each other")
        if all_borders[len(all_borders) - 1][1] != size:
            raise ValueError("Borders don't match with size")
        return True

    def check_materials(self, T: festim.Temperature, derived_quantities: list = []):
        """Checks the materials keys

        Args:
            T (festim.Temperature): the temperature
            derived_quantities (list): list of festim.DerivedQuantity
                objects the derived quantities. Defaults to [].
        """

        if len(self.materials) > 0:  # TODO: get rid of this...
            self.check_consistency()

            self.check_for_unused_properties(T, derived_quantities)

            self.check_unique_ids()

            self.check_missing_properties(T, derived_quantities)

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

    def check_for_unused_properties(
        self, T: festim.Temperature, derived_quantities: list
    ):
        """Warns users if properties will be ignored

        Args:
            T (festim.Temperature): the temperature
            derived_quantities (list): list of festim.DerivedQuantity
                objects
        """
        # TODO add a check for ignored solubility when chemical_pot is False
        # warn about unused keys
        transient_properties = ["rho", "heat_capacity"]
        if not isinstance(T, HeatTransferProblem):
            for mat in self.materials:
                for key in transient_properties:
                    if getattr(mat, key) is not None:
                        warnings.warn(key + " key will be ignored", UserWarning)

        for mat in self.materials:
            if getattr(mat, "thermal_cond") is not None:
                warn = True
                if isinstance(T, HeatTransferProblem):
                    warn = False
                else:
                    surface_fluxes = list(
                        quant
                        for quant in derived_quantities
                        if isinstance(quant, festim.SurfaceFlux)
                    )

                    for surface_flux in surface_fluxes:
                        if surface_flux.field == "T":
                            warn = False
                if warn:
                    warnings.warn("thermal_cond key will be ignored", UserWarning)

    def check_consistency(self):
        """Checks that materials have the same attributes"""
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

    def check_missing_properties(self, T: festim.Temperature, derived_quantities: list):
        """Checks if the materials miss some properties

        Args:
            T (festim.Temperature): the temperature
            derived_quantities (list): list of festim.DerivedQuantity objects

        Raises:
            ValueError: if thermal_cond, heat_capacity or rho is None when needed
        """
        if isinstance(T, HeatTransferProblem):
            if self.materials[0].thermal_cond is None:
                raise ValueError("Missing thermal_cond in materials")
            if T.transient:
                if self.materials[0].heat_capacity is None:
                    raise ValueError("Missing heat_capacity in materials")
                if self.materials[0].rho is None:
                    raise ValueError("Missing rho in materials")
        # TODO: add check for thermal cond for thermal flux computation

    def find_material_from_id(self, mat_id):
        """Returns the material from a given id

        Args:
            mat_id (int): id of the wanted material

        Raises:
            ValueError: if the id isn't found

        Returns:
            festim.Material: the material that has the id mat_id
        """
        for material in self.materials:
            mat_ids = material.id
            if type(mat_ids) is not list:
                mat_ids = [mat_ids]
            if mat_id in mat_ids:
                return material
        raise ValueError("Couldn't find ID " + str(mat_id) + " in materials list")

    def find_material_from_name(self, name):
        """Returns the material with the correct name

        Args:
            name (str): the name of the material

        Raises:
            ValueError: when no match was found

        Returns:
            festim.Material: the material object
        """
        for material in self.materials:
            if material.name == name:
                return material

        msg = "No material with name {} was found".format(name)
        raise ValueError(msg)

    def find_material(self, mat: Union[int, str, Material]):
        """Returns the correct festim.Material object based on either an id,
        a name

        Args:
            mat (Union[int, str, Material]): the material tag

        Returns:
            festim.Material: the matching material
        """
        if isinstance(mat, int):
            return self.find_material_from_id(mat)
        elif isinstance(mat, str):
            return self.find_material_from_name(mat)
        elif isinstance(mat, Material):
            return mat

    def find_subdomain_from_x_coordinate(self, x):
        """Finds the correct subdomain at a given x coordinate

        Args:
            x (float): the x coordinate

        Returns:
            int: the corresponding subdomain id
        """
        for material in self.materials:
            # if no borders are provided, assume only one subdomain
            if material.borders is None:
                return material.id
            # else find the correct material
            else:
                if isinstance(material.borders[0], list) and len(material.borders) > 1:
                    list_of_borders = material.borders
                else:
                    list_of_borders = [material.borders]
                if isinstance(material.id, list):
                    subdomains = material.id
                else:
                    subdomains = [material.id for _ in range(len(list_of_borders))]

                for borders, subdomain in zip(list_of_borders, subdomains):
                    if borders[0] <= x <= borders[1]:
                        return subdomain
        # if no subdomain was found, return 0
        return 0

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
            self.thermal_cond = ThermalProp(self, vm, T, "thermal_cond", degree=2)
            self.heat_capacity = ThermalProp(self, vm, T, "heat_capacity", degree=2)
            self.density = ThermalProp(self, vm, T, "rho", degree=2)
        if self.materials[0].H is not None:
            self.H = HCoeff(self, vm, T, degree=2)

    def solubility_as_function(self, mesh, T):
        """
        Makes solubility as a fenics.Function and stores it in S attribute
        """
        V = f.FunctionSpace(mesh.mesh, "DG", 1)
        S = f.Function(V, name="S")
        vS = f.TestFunction(V)
        dx = mesh.dx
        F = 0
        for mat in self.materials:
            F += -S * vS * dx(mat.id)
            F += mat.S_0 * f.exp(-mat.E_S / k_B / T) * vS * dx(mat.id)
        f.solve(F == 0, S, bcs=[])

        self.S = S

    def create_solubility_law_markers(self, mesh: festim.Mesh):
        """Creates the attributes henry_marker and sievert_marker
        These fenics.Function are equal to one or zero depending
        on the material solubility_law

        Args:
            mesh (festim.Mesh): the mesh
        """
        V = f.FunctionSpace(mesh.mesh, "DG", 0)
        henry = f.Function(V)
        sievert = f.Function(V)

        test_function_henry = f.TestFunction(V)
        test_function_sievert = f.TestFunction(V)

        # initialise formulations
        F_henry = -henry * test_function_henry * mesh.dx
        F_sievert = -sievert * test_function_sievert * mesh.dx

        # build the formulation depending on the
        for mat in self.materials:
            # make sure mat_ids is a list
            mat_ids = mat.id
            if not isinstance(mat.id, list):
                mat_ids = [mat.id]

            for mat_id in mat_ids:  # iterate through the subdomains
                if mat.solubility_law == "henry":
                    F_henry += 1 * test_function_henry * mesh.dx(mat_id)
                elif mat.solubility_law == "sievert":
                    F_sievert += 1 * test_function_sievert * mesh.dx(mat_id)

        # solve the problems
        f.solve(F_henry == 0, henry, [])
        f.solve(F_sievert == 0, sievert, [])

        self.henry_marker = henry
        self.sievert_marker = sievert


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
        value[0] = D_0 * f.exp(-E_D / k_B / self._T(x))

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

        value[0] = material.free_enthalpy + self._T(x) * material.entropy

    def value_shape(self):
        return ()
