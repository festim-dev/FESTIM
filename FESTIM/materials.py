from operator import itemgetter
import warnings
import numpy as np


class Material:
    def __init__(self, id, D_0, E_D, S_0=None, E_S=None, thermal_cond=None, heat_capacity=None, rho=None, borders=[], H=None) -> None:
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
        if self.S_0 is None and self.E_S is not None:
            raise ValueError("S_0 cannot be None")
        if self.E_S is None and self.S_0 is not None:
            raise ValueError("E_S cannot be None")


class Materials:
    def __init__(self, materials=[]):
        self.materials = materials

    def check_borders(self, size, materials):
        """Checks that the borders given match

        Arguments:
            size {float} -- float, size of the domain
            materials {list} -- contains dicts with materials parameters

        Raises:
            ValueError: if the borders don't begin at zero
            ValueError: if borders don't match
            ValueError: if borders don't end at size

        Returns:
            bool -- True if everything's alright
        """
        all_borders = []
        for m in materials:
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

        self.check_consistency()

        self.check_for_unused_properties(temp_type, derived_quantities)

        self.check_unique_ids()

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
