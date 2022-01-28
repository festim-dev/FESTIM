import fenics as f
import FESTIM


def is_export_xdmf(export, t, final_time, nb_iterations):
    """Checks if export should be exported.

    Args:
        export (FESTIM.XDMFExport): the export object
        t (float): the current time
        final_time (float): the final time of the simulation
        nb_iterations (int): the current number of time steps

    Returns:
        bool: True if export should be exported, else False
    """
    # TODO this should be a method of XDMFExport
    if (export.last_time_step_only and
        t >= final_time) or \
            not export.last_time_step_only:
        if nb_iterations % \
                export.nb_iterations_between_exports == 0:
            return True

    return False


def is_export_derived_quantities(derived_quantities, t, final_time, nb_iterations):
    """Checks if the derived quantities should be exported or not based on the
    key simulation.nb_iterations_between_export_derived_quantities

    Args:
        derived_quantities (FESTIM.DerivedQuantities): the derived quantities
        t (float): the current time
        final_time (float): the final time of the simulation
        nb_iterations (int): the current number of time steps

    Returns:
        bool: True if the derived quantities should be exported, else False
    """
    # TODO this should be a method of DerivedQuantities
    if final_time is not None:
        nb_its_between_exports = \
            derived_quantities.nb_iterations_between_exports
        if nb_its_between_exports is None:
            # export at the end
            return t >= final_time
        else:
            # export every N iterations
            return nb_iterations % nb_its_between_exports == 0
    else:
        # if steady state, export
        return True


class ArheniusCoeff(f.UserExpression):
    def __init__(self, mesh, materials, vm, T, pre_exp, E, **kwargs):
        super().__init__(kwargs)
        self._mesh = mesh
        self._vm = vm
        self._T = T
        self._materials = materials
        self._pre_exp = pre_exp
        self._E = E

    def eval_cell(self, value, x, ufc_cell):
        cell = f.Cell(self._mesh, ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = self._materials.find_material_from_id(subdomain_id)
        D_0 = getattr(material, self._pre_exp)
        E_D = getattr(material, self._E)
        value[0] = D_0*f.exp(-E_D/FESTIM.k_B/self._T(x))

    def value_shape(self):
        return ()


class ThermalProp(f.UserExpression):
    def __init__(self, mesh, materials, vm, T, key, **kwargs):
        super().__init__(kwargs)
        self._mesh = mesh
        self._T = T
        self._vm = vm
        self._materials = materials
        self._key = key

    def eval_cell(self, value, x, ufc_cell):
        cell = f.Cell(self._mesh, ufc_cell.index)
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
    def __init__(self, mesh, materials, vm, T, **kwargs):
        super().__init__(kwargs)
        self._mesh = mesh
        self._T = T
        self._vm = vm
        self._materials = materials

    def eval_cell(self, value, x, ufc_cell):
        cell = f.Cell(self._mesh, ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = self._materials.find_material_from_id(subdomain_id)

        value[0] = material.free_enthalpy + \
            self._T(x)*material.entropy

    def value_shape(self):
        return ()


def create_properties(mesh, materials, vm, T):
    """Creates the properties fields needed for post processing

    Arguments:
        mesh {fenics.Mesh()} -- the mesh
        materials {FESTIM.Materials} -- contains materials parameters
        vm {fenics.MeshFunction()} -- volume markers
        T {fenics.Function()} -- temperature

    Returns:
        ArheniusCoeff -- diffusion coefficient (SI)
        ThermalProp -- thermal conductivity (SI)
        ThermalProp -- heat capactiy (SI)
        ThermalProp -- density (kg/m3)
        HCoeff -- enthalpy (SI)
        ArheniusCoeff -- solubility coefficient (SI)
    """
    # TODO: this could be refactored since vm contains the mesh
    D = ArheniusCoeff(mesh, materials, vm, T, "D_0", "E_D", degree=2)
    thermal_cond = None
    cp = None
    rho = None
    H = None
    S = None
    # all materials have the same properties so only checking the first is enough
    if materials.materials[0].S_0 is not None:
        S = ArheniusCoeff(mesh, materials, vm, T, "S_0", "E_S", degree=2)
    if materials.materials[0].thermal_cond is not None:
        thermal_cond = ThermalProp(mesh, materials, vm, T,
                                    'thermal_cond', degree=2)
        cp = ThermalProp(mesh, materials, vm, T,
                            'heat_capacity', degree=2)
        rho = ThermalProp(mesh, materials, vm, T,
                            'rho', degree=2)
    if materials.materials[0].H is not None:
        H = HCoeff(mesh, materials, vm, T, degree=2)

    return D, thermal_cond, cp, rho, H, S
