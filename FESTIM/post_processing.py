from fenics import *
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
    T = simulation.T
    markers = [simulation.volume_markers, simulation.surface_markers]
    V_DG1, V_CG1 = simulation.V_DG1, simulation.V_CG1
    t = simulation.t
    dt = simulation.dt
    files = simulation.files
    append = simulation.append
    D, thermal_cond, cp, rho, H, S = \
        simulation.D, simulation.thermal_cond, simulation.cp, simulation.rho, \
        simulation.H, simulation.S
    derived_quantities_global = simulation.derived_quantities_global

    if not append:
        if "derived_quantities" in parameters["exports"].keys():
            derived_quantities_global.append(
                FESTIM.post_processing.header_derived_quantities(parameters))

    if u.function_space().num_sub_spaces() == 0:
        res = [u]
    else:
        res = list(u.split())
    if simulation.chemical_pot:
        solute = res[0]*S
        res[0] = solute

    retention = sum(res)
    res.append(retention)
    res.append(T)

    if "derived_quantities" in parameters["exports"].keys():

        if is_compute_derived_quantities(simulation):
            derived_quantities_t = \
                FESTIM.post_processing.derived_quantities(
                    parameters,
                    res,
                    markers,
                    [D, thermal_cond, H]
                    )
            derived_quantities_t.insert(0, t)
            derived_quantities_global.append(derived_quantities_t)
        if is_export_derived_quantities(simulation):
            FESTIM.write_to_csv(
                simulation.parameters["exports"]["derived_quantities"],
                simulation.derived_quantities_global)

    if "xdmf" in parameters["exports"].keys():
        if (simulation.export_xdmf_last_only and
            simulation.t >= simulation.final_time) or \
                not simulation.export_xdmf_last_only:
            if simulation.nb_iterations % \
                    simulation.nb_iterations_between_exports == 0:
                functions_to_exports = \
                    parameters["exports"]["xdmf"]["functions"]
                # if solute or retention needs to be exported,
                # project it onto V_DG1
                if any(x in functions_to_exports for x in ['0', 'solute']):
                    if simulation.chemical_pot:
                        # this is costly ...
                        res[0] = project(res[0], V_DG1)
                if 'retention' in functions_to_exports:
                    res[-2] = project(retention, V_DG1)

                FESTIM.export.export_xdmf(
                    res, parameters["exports"], files, t, append=append)
    if "txt" in parameters["exports"].keys():
        dt = FESTIM.export.export_profiles(
            res, parameters["exports"], t, dt, V_DG1)

    return derived_quantities_global, dt


def is_compute_derived_quantities(simulation):
    """Checks if the derived quantities should be computed or not based on the
    key "nb_iterations_between_compute"

    Args:
        simulation (FESTIM.Simulation): the main Simulation instance

    Returns:
        bool: True if the derived quantities should be computed, else False
    """
    if simulation.nb_iterations % \
            simulation.nb_iterations_between_compute_derived_quantities == 0:
        return True
    return False


def is_export_derived_quantities(simulation):
    """Checks if the derived quantities should be exported or not based on the
    key "nb_iterations_between_export"

    Args:
        simulation (FESTIM.Simulation): the main Simulation instance

    Returns:
        bool: True if the derived quantities should be exported, else False
    """
    if simulation.transient:
        if (simulation.nb_iterations_between_export_derived_quantities is None
                and simulation.t >= simulation.final_time):
            return True
        elif simulation.nb_iterations % \
                simulation.nb_iterations_between_export_derived_quantities == 0:
            return True
        else:
            return False
    else:
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
            exact_sol = Expression(sp.printing.ccode(
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
                error_L2 = errornorm(
                    exact_sol, computed_sol, error["norm"])
                er.append(error_L2)

        tab.append(er)
    return tab


class ArheniusCoeff(UserExpression):
    def __init__(self, mesh, materials, vm, T, pre_exp, E, **kwargs):
        super().__init__(kwargs)
        self._mesh = mesh
        self._vm = vm
        self._T = T
        self._materials = materials
        self._pre_exp = pre_exp
        self._E = E

    def eval_cell(self, value, x, ufc_cell):
        cell = Cell(self._mesh, ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = FESTIM.helpers.find_material_from_id(
            self._materials, subdomain_id)
        D_0 = material[self._pre_exp]
        E_D = material[self._E]
        value[0] = D_0*exp(-E_D/FESTIM.k_B/self._T(x))

    def value_shape(self):
        return ()


class ThermalProp(UserExpression):
    def __init__(self, mesh, materials, vm, T, key, **kwargs):
        super().__init__(kwargs)
        self._mesh = mesh
        self._T = T
        self._vm = vm
        self._materials = materials
        self._key = key

    def eval_cell(self, value, x, ufc_cell):
        cell = Cell(self._mesh, ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = FESTIM.helpers.find_material_from_id(
            self._materials, subdomain_id)
        if callable(material[self._key]):
            value[0] = material[self._key](self._T(x))
        else:
            value[0] = material[self._key]

    def value_shape(self):
        return ()


class HCoeff(UserExpression):
    def __init__(self, mesh, materials, vm, T, **kwargs):
        super().__init__(kwargs)
        self._mesh = mesh
        self._T = T
        self._vm = vm
        self._materials = materials

    def eval_cell(self, value, x, ufc_cell):
        cell = Cell(self._mesh, ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = FESTIM.helpers.find_material_from_id(
            self._materials, subdomain_id)

        value[0] = material["H"]["free_enthalpy"] + \
            self._T(x)*material["H"]["entropy"]

    def value_shape(self):
        return ()


def create_properties(mesh, materials, vm, T):
    """Creates the properties fields needed for post processing

    Arguments:
        mesh {fenics.Mesh()} -- the mesh
        materials {dict} -- contains materials parameters
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

    D = ArheniusCoeff(mesh, materials, vm, T, "D_0", "E_D", degree=2)
    thermal_cond = None
    cp = None
    rho = None
    H = None
    S = None
    for mat in materials:
        if "S_0" in mat.keys():
            S = ArheniusCoeff(mesh, materials, vm, T, "S_0", "E_S", degree=2)
        if "thermal_cond" in mat.keys():
            thermal_cond = ThermalProp(mesh, materials, vm, T,
                                       'thermal_cond', degree=2)
            cp = ThermalProp(mesh, materials, vm, T,
                             'heat_capacity', degree=2)
            rho = ThermalProp(mesh, materials, vm, T,
                              'rho', degree=2)
        if "H" in mat.keys():
            H = HCoeff(mesh, materials, vm, T, degree=2)

    return D, thermal_cond, cp, rho, H, S


def calculate_maximum_volume(f, subdomains, subd_id):
    '''Minimum of f over subdomains cells marked with subd_id'''
    V = f.function_space()

    dm = V.dofmap()

    subd_dofs = np.unique(np.hstack(
        [dm.cell_dofs(c.index())
         for c in SubsetIterator(subdomains, subd_id)]))

    return np.max(f.vector().get_local()[subd_dofs])


def calculate_minimum_volume(f, subdomains, subd_id):
    '''Minimum of f over subdomains cells marked with subd_id'''
    V = f.function_space()

    dm = V.dofmap()

    subd_dofs = np.unique(np.hstack(
        [dm.cell_dofs(c.index())
         for c in SubsetIterator(subdomains, subd_id)]))

    return np.min(f.vector().get_local()[subd_dofs])


def header_derived_quantities(parameters):
    '''
    Creates the header for derived_quantities list
    '''

    check_keys_derived_quantities(parameters)
    derived_quant_dict = parameters["exports"]["derived_quantities"]
    header = ['t(s)']
    if "surface_flux" in derived_quant_dict.keys():
        for flux in derived_quant_dict["surface_flux"]:
            for surf in flux["surfaces"]:
                header.append(
                    "Flux surface " + str(surf) + ": " + str(flux['field']))
    if "average_volume" in derived_quant_dict.keys():
        for average in parameters[
                        "exports"]["derived_quantities"]["average_volume"]:
            for vol in average["volumes"]:
                header.append(
                    "Average " + str(average['field']) + " volume " + str(vol))
    if "minimum_volume" in derived_quant_dict.keys():
        for minimum in parameters[
                        "exports"]["derived_quantities"]["minimum_volume"]:
            for vol in minimum["volumes"]:
                header.append(
                    "Minimum " + str(minimum["field"]) + " volume " + str(vol))
    if "maximum_volume" in derived_quant_dict.keys():
        for maximum in parameters[
                        "exports"]["derived_quantities"]["maximum_volume"]:
            for vol in maximum["volumes"]:
                header.append(
                    "Maximum " + str(maximum["field"]) + " volume " + str(vol))
    if "total_volume" in derived_quant_dict.keys():
        for total in derived_quant_dict["total_volume"]:
            for vol in total["volumes"]:
                header.append(
                    "Total " + str(total["field"]) + " volume " + str(vol))
    if "total_surface" in derived_quant_dict.keys():
        for total in derived_quant_dict["total_surface"]:
            for surf in total["surfaces"]:
                header.append(
                    "Total " + str(total["field"]) + " surface " + str(surf))

    return header


def derived_quantities(parameters, solutions,
                       markers, properties):
    """Computes all the derived_quantities and stores it into list

    Arguments:
        parameters {dict} -- main parameters dict
        solutions {list} -- contains fenics.Function
        markers {list} -- contains volume and surface markers
        properties {list} -- contains properties
            [D, thermal_cond, cp, rho, H, S]

    Returns:
        list -- list of derived quantities
    """

    D = properties[0]
    thermal_cond = properties[1]
    soret = False
    if "temperature" in parameters.keys():
        if "soret" in parameters["temperature"].keys():
            if parameters["temperature"]["soret"] is True:
                soret = True
                Q = properties[2]
    volume_markers = markers[0]
    surface_markers = markers[1]
    mesh = solutions[-1].function_space().mesh()
    n = FacetNormal(mesh)
    dx = Measure('dx', domain=mesh, subdomain_data=volume_markers)
    ds = Measure('ds', domain=mesh, subdomain_data=surface_markers)

    # Create dicts

    ret = solutions[len(solutions)-2]
    V_DG1 = FunctionSpace(mesh, "DG", 1)

    T = solutions[len(solutions)-1]
    field_to_sol = {
        'solute': solutions[0],
        'retention': ret,
        'T': T,
    }
    field_to_prop = {
        'solute': D,
        'T': thermal_cond,
    }
    for i in range(1, len(solutions)-2):
        field_to_sol[str(i)] = solutions[i]

    tab = []
    # Compute quantities
    derived_quant_dict = parameters["exports"]["derived_quantities"]
    if "surface_flux" in derived_quant_dict.keys():
        for flux in derived_quant_dict["surface_flux"]:
            sol = field_to_sol[str(flux["field"])]
            # TODO: find an alternative for this is costly
            if isinstance(sol, Product):
                sol = project(sol, V_DG1)
            prop = field_to_prop[str(flux["field"])]
            for surf in flux["surfaces"]:
                phi = assemble(prop*dot(grad(sol), n)*ds(surf))
                if soret is True and str(flux["field"]) == 'solute':
                    phi += assemble(
                        prop*sol*Q/(FESTIM.R*T**2)*dot(grad(T), n)*ds(surf))
                tab.append(phi)
    if "average_volume" in derived_quant_dict.keys():
        for average in parameters[
                        "exports"]["derived_quantities"]["average_volume"]:
            sol = field_to_sol[str(average["field"])]
            for vol in average["volumes"]:
                val = assemble(sol*dx(vol))/assemble(1*dx(vol))
                tab.append(val)
    if "minimum_volume" in derived_quant_dict.keys():
        for minimum in parameters[
                        "exports"]["derived_quantities"]["minimum_volume"]:
            if str(minimum["field"]) == "retention":
                for vol in minimum["volumes"]:
                    val = 0
                    for f in solutions[0:-2]:
                        val += calculate_minimum_volume(f, volume_markers, vol)
                    tab.append(val)
            else:
                sol = field_to_sol[str(minimum["field"])]
                for vol in minimum["volumes"]:
                    tab.append(calculate_minimum_volume(
                        sol, volume_markers, vol))
    if "maximum_volume" in derived_quant_dict.keys():
        for maximum in parameters[
                        "exports"]["derived_quantities"]["maximum_volume"]:
            if str(maximum["field"]) == "retention":
                for vol in maximum["volumes"]:
                    val = 0
                    for f in solutions[0:-2]:
                        val += calculate_maximum_volume(f, volume_markers, vol)
                    tab.append(val)
            else:
                sol = field_to_sol[str(maximum["field"])]
                for vol in maximum["volumes"]:
                    tab.append(calculate_maximum_volume(
                        sol, volume_markers, vol))
    if "total_volume" in derived_quant_dict.keys():
        for total in derived_quant_dict["total_volume"]:
            sol = field_to_sol[str(total["field"])]
            for vol in total["volumes"]:
                tab.append(assemble(sol*dx(vol)))
    if "total_surface" in derived_quant_dict.keys():
        for total in derived_quant_dict["total_surface"]:
            sol = field_to_sol[str(total["field"])]
            for surf in total["surfaces"]:
                tab.append(assemble(sol*ds(surf)))
    return tab


def check_keys_derived_quantities(parameters):
    """Checks the keys in derived quantities dict

    Arguments:
        parameters {dict} -- main parameters dict

    Raises:
        ValueError: if quantity is unknown
        KeyError: if the field key is missing
        ValueError: if a field is unknown
        ValueError: if a field is unknown
        ValueError: if a field is unknown
        KeyError: if surfaces or volumes key is missing
    """
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
                        if f["field"] > len(parameters["traps"]) or \
                           f["field"] < 0:
                            raise ValueError(
                                "Unknown field: " + str(f["field"]))
                    elif type(f["field"]) is str:
                        if f["field"] not in FESTIM.helpers.field_types:
                            if f["field"].isdigit():
                                if int(f["field"]) > len(parameters["traps"]) \
                                    or \
                                   int(f["field"]) < 0:
                                    raise ValueError(
                                        "Unknown field: " + f["field"])
                            else:
                                raise ValueError(
                                    "Unknown field: " + str(f["field"]))

                if "surfaces" not in f.keys() and "volumes" not in f.keys():
                    raise KeyError("Missing key 'surfaces' or 'volumes'")
