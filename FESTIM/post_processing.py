import fenics as f
from ufl.algebra import Product
import sympy as sp
import numpy as np
import FESTIM


def run_post_processing(simulation):
    """Main post processing FESTIM function.

    Arguments:

    Returns:
        list -- updated derived quantities list
        fenics.Constant() -- updated stepsize
    """
    parameters = simulation.parameters
    transient = simulation.transient
    u = simulation.u
    T = simulation.T.T
    markers = [simulation.volume_markers, simulation.surface_markers]
    V_DG1, V_CG1 = simulation.V_DG1, simulation.V_CG1
    t = simulation.t
    dt = simulation.dt
    files = simulation.files
    D, thermal_cond, cp, rho, H, S = \
        simulation.D, simulation.thermal_cond, simulation.cp, simulation.rho, \
        simulation.H, simulation.S

    label_to_function = {
        "solute": simulation.mobile.post_processing_solution,
        "0": simulation.mobile.post_processing_solution,
        0: simulation.mobile.post_processing_solution,
        "T": simulation.T.T,
        "retention": sum([simulation.mobile.post_processing_solution] + [trap.post_processing_solution for trap in simulation.traps.traps])
    }
    for trap in simulation.traps.traps:
        label_to_function[trap.id] = trap.post_processing_solution
        label_to_function[str(trap.id)] = trap.post_processing_solution

    # make the change of variable solute = theta*S
    if simulation.chemical_pot:
        label_to_function["solute"] = simulation.mobile.solution*S  # solute = theta*S = (solute/S) * S

        if simulation.need_projecting_solute():
            # project solute on V_DG1
            label_to_function["solute"] = f.project(label_to_function["solute"], V_DG1)

    for export in simulation.exports.exports:
        if isinstance(export, FESTIM.DerivedQuantities):
            # compute derived quantities
            if simulation.nb_iterations % export.nb_iterations_between_compute == 0:
                export.compute(t, label_to_function)
            # export derived quantities
            if is_export_derived_quantities(simulation, export):
                export.write()

        elif isinstance(export, FESTIM.XDMFExport):
            if (export.last_time_step_only and
                simulation.t >= simulation.final_time) or \
                    not export.last_time_step_only:
                if simulation.nb_iterations % \
                        export.nb_iterations_between_exports == 0:
                    if export.function == "retention":
                        # if not a Function, project it onto V_DG1
                        if not isinstance(label_to_function["retention"], f.Function):
                            label_to_function["retention"] = f.project(label_to_function["retention"], V_DG1)
                    export.write(label_to_function, simulation.t)
                    export.append = True

        elif isinstance(export, FESTIM.TXTExport):
            export.write(label_to_function, t, dt)

    return dt


def is_export_derived_quantities(simulation, derived_quantities):
    """Checks if the derived quantities should be exported or not based on the
    key simulation.nb_iterations_between_export_derived_quantities

    Args:
        simulation (FESTIM.Simulation): the main Simulation instance
        derived_quantities (FESTIM.DerivedQuantities): the derived quantities

    Returns:
        bool: True if the derived quantities should be exported, else False
    """
    if simulation.transient:
        nb_its_between_exports = \
            derived_quantities.nb_iterations_between_exports
        if nb_its_between_exports is None:
            # export at the end
            return simulation.t >= simulation.final_time
        else:
            # export every N iterations
            return simulation.nb_iterations % nb_its_between_exports == 0
    else:
        # if steady state, export
        return True


def compute_error(parameters, t, res, mesh):
    """Returns a list containing the errors

    Arguments:
        parameters {dict} -- error parameters dict
        t {float} -- time
        res {list} -- contains the solutions
        mesh {fenics.Mesh()} -- the mesh

    Raises:
        KeyError: if key is not found in dict

    Returns:
        list -- list of errors
    """
    tab = []

    solution_dict = {
        'solute': res[0],
        'retention': res[len(res)-2],
        'T': res[len(res)-1],
    }

    for error in parameters:
        er = []
        er.append(t)
        for i in range(len(error["exact_solutions"])):
            exact_sol = f.Expression(sp.printing.ccode(
                error["exact_solutions"][i]),
                degree=error["degree"],
                t=t)
            err = error["computed_solutions"][i]
            if type(err) is str:
                if err.isdigit():
                    nb = int(err)
                    computed_sol = res[nb]
                else:
                    if err in solution_dict.keys():
                        computed_sol = solution_dict[
                            err]
                    else:
                        raise KeyError(
                            "The key " + err + " is unknown.")
            elif type(err) is int:
                computed_sol = res[err]

            if error["norm"] == "error_max":
                vertex_values_u = computed_sol.compute_vertex_values(mesh)
                vertex_values_sol = exact_sol.compute_vertex_values(mesh)
                error_max = np.max(np.abs(vertex_values_u - vertex_values_sol))
                er.append(error_max)
            else:
                error_L2 = f.errornorm(
                    exact_sol, computed_sol, error["norm"])
                er.append(error_L2)

        tab.append(er)
    return tab


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


def check_keys_derived_quantities(simulation):
    """Checks the keys in derived quantities dict

    Raises:
        ValueError: if quantity is unknown
        KeyError: if the field key is missing
        ValueError: if a field is unknown
        ValueError: if a field is unknown
        ValueError: if a field is unknown
        KeyError: if surfaces or volumes key is missing
    """
    parameters = simulation.parameters
    for quantity in parameters["exports"]["derived_quantities"].keys():
        non_quantity_types = [
            "file", "folder",
            "nb_iterations_between_compute", "nb_iterations_between_exports"]
        if quantity not in \
                [*FESTIM.helpers.quantity_types] + non_quantity_types:
            raise ValueError("Unknown quantity: " + quantity)
        if quantity not in non_quantity_types:
            for f in parameters["exports"]["derived_quantities"][quantity]:
                if "field" not in f.keys():
                    raise KeyError("Missing key 'field'")
                else:
                    if type(f["field"]) is int:
                        if f["field"] > len(simulation.traps.traps) or \
                           f["field"] < 0:
                            raise ValueError(
                                "Unknown field: " + str(f["field"]))
                    elif type(f["field"]) is str:
                        if f["field"] not in FESTIM.helpers.field_types:
                            if f["field"].isdigit():
                                if simulation.traps.traps != []:
                                    if int(f["field"]) > len(simulation.traps.traps) \
                                        or \
                                        int(f["field"]) < 0:
                                        raise ValueError(
                                            "Unknown field: " + f["field"])
                                else:
                                    raise ValueError(
                                            "Unknown field: " + f["field"])
                            else:
                                raise ValueError(
                                    "Unknown field: " + str(f["field"]))

                if "surfaces" not in f.keys() and "volumes" not in f.keys():
                    raise KeyError("Missing key 'surfaces' or 'volumes'")
