from fenics import *
from ufl.algebra import Product
import sympy as sp
import numpy as np
import FESTIM


def run_post_processing(simulation):
    """Main post processing FESTIM function.

    Args:
        simulation (FESTIM.Simulation()): Simulation object

    Returns:
        list, fenics.Constant(): derived quantities list, stepsize
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
        derived_quantities_t = \
            FESTIM.post_processing.derived_quantities(
                parameters,
                res,
                markers,
                [D, thermal_cond, H]
                )

        derived_quantities_t.insert(0, t)
        derived_quantities_global.append(derived_quantities_t)
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


def compute_error(errors_dict, t, res, mesh):
    """Returns a list containing the errors

    Args:
        errors_dict (list): list of dicts
            {"exact_solutions: ...,
              "computed_solutions": ...,
              "degree": ...,
              "norm": ...}
            The value of "exact_solutions" is a list of sympy expressions
            The value of "computed_solutions" is a list of sympy expressions
            The value of "degree" is an int
            The value of "norm" can be "error_max" or "L2"
        t (float): time (s)
        res (list): list of fenics.Function()
            [solute, trap1, trap2, ..., retention, temperature]
        mesh (fenics.Mesh()): simulation mesh

    Raises:
        KeyError: if key is not found in dict

    Returns:
        list: list of lists of floats containing the errors
    """
    tab = []

    solution_dict = {
        'solute': res[0],
        'retention': res[len(res)-2],
        'T': res[len(res)-1],
    }

    for error in errors_dict:
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
    """Expression for an Arhenius coefficient

    Args:
        mesh (fenics.Mesh()): the mesh
        materials (list): list of dicts
        vm (fenics.MeshFunction()): mesh function containing tags for
            subdomains
        T (fenics.Function()): Temperature (K)
        pre_exp (str): key corresponding to the pre-exponential factor of the
            coefficient in the material dict
        E (str): key corresponding to the activation energy of the
            coefficient in the material dict
    """
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
    """Expression for a thermal property

    Args:
        mesh (fenics.Mesh()): the mesh
        materials (list): list of dicts
        vm (fenics.MeshFunction()): mesh function containing tags for
            subdomains
        T (fenics.Function()): Temperature (K)
        key (str): key corresponding to the property in the material dict
    """

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
    """Expression for the heat of transport

    Args:
        mesh (fenics.Mesh()): the mesh
        materials (list): list of dicts
        vm (fenics.MeshFunction()): mesh function containing tags for
            subdomains
        T (fenics.Function()): Temperature (K)
    """
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

    Args:
        mesh (fenics.Mesh()): the mesh
        materials (list): list of dicts
        vm (fenics.MeshFunction()): mesh function containing tags for
            subdomains
        T (fenics.Function()): Temperature (K)

    Returns:
        tuple of UserExpression: Diffusion coefficient, thermal conductivity,
            heat capacity, density, heat of transport, solubility
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
    """Maximum of a function f on a given subdomain

    Args:
        f (fenics.Function()): the function
        subdomains (fenics.MeshFunction()): MeshFunction containing the physical entities
        subd_id (int): The tag of the subdomain

    Returns:
        float: the maximum value of f over the subdomain subd_id
    """
    V = f.function_space()

    dm = V.dofmap()

    subd_dofs = np.unique(np.hstack(
        [dm.cell_dofs(c.index())
         for c in SubsetIterator(subdomains, subd_id)]))

    return np.max(f.vector().get_local()[subd_dofs])


def calculate_minimum_volume(f, subdomains, subd_id):
    """Minimum of a function f on a given subdomain

    Args:
        f (fenics.Function()): the function
        subdomains (fenics.MeshFunction()): MeshFunction containing the physical entities
        subd_id (int): The tag of the subdomain

    Returns:
        float: the minimum value of f over the subdomain subd_id
    """
    V = f.function_space()

    dm = V.dofmap()

    subd_dofs = np.unique(np.hstack(
        [dm.cell_dofs(c.index())
         for c in SubsetIterator(subdomains, subd_id)]))

    return np.min(f.vector().get_local()[subd_dofs])


def header_derived_quantities(parameters):
    """Creates the header for derived_quantities list

    Args:
        parameters (dict): maint parameters dict

    Returns:
        list: header of the derived quantities list
    """

    check_keys_derived_quantities(parameters)
    derived_quant_dict = parameters["exports"]["derived_quantities"]
    header = ['t(s)']
    if "surface_flux" in derived_quant_dict.keys():
        for flux in derived_quant_dict["surface_flux"]:
            for id_subdomain in flux["surfaces"]:
                header.append(
                    "Flux surface " + str(id_subdomain) + ": " + str(flux['field']))
    if "average_volume" in derived_quant_dict.keys():
        for average in derived_quant_dict["average_volume"]:
            for id_subdomain in average["volumes"]:
                header.append(
                    "Average " + str(average['field']) + " volume " + str(id_subdomain))
    if "minimum_volume" in derived_quant_dict.keys():
        for minimum in derived_quant_dict["minimum_volume"]:
            for id_subdomain in minimum["volumes"]:
                header.append(
                    "Minimum " + str(minimum["field"]) + " volume " + str(id_subdomain))
    if "maximum_volume" in derived_quant_dict.keys():
        for maximum in derived_quant_dict["maximum_volume"]:
            for id_subdomain in maximum["volumes"]:
                header.append(
                    "Maximum " + str(maximum["field"]) + " volume " + str(id_subdomain))
    if "total_volume" in derived_quant_dict.keys():
        for total in derived_quant_dict["total_volume"]:
            for id_subdomain in total["volumes"]:
                header.append(
                    "Total " + str(total["field"]) + " volume " + str(id_subdomain))
    if "total_surface" in derived_quant_dict.keys():
        for total in derived_quant_dict["total_surface"]:
            for id_subdomain in total["surfaces"]:
                header.append(
                    "Total " + str(total["field"]) + " surface " + str(id_subdomain))

    return header


def derived_quantities(parameters, solutions,
                       markers, properties):
    """Computes all the derived_quantities and stores it into list

    Args:
        parameters (dict): main parameters dict
        solutions (list): contains fenics.Function
        markers ((fenics.MeshFunction, fenics.MeshFunction)):
            (volume markers, surface markers)
        properties (list of fenics.Function): [D, thermal_cond, cp, rho, H, S]

    Returns:
        list: list of derived quantities
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

    Args:
        parameters ([type]): [description]

    Raises:
        ValueError: if quantity is unknown
        KeyError: if the field key is missing
        ValueError: if a field is unknown
        KeyError: if surfaces or volumes key is missing
    """
    derived_quantities = parameters["exports"]["derived_quantities"]
    for quantity in derived_quantities.keys():
        if quantity not in [*FESTIM.helpers.quantity_types, "file", "folder"]:
            raise ValueError("Unknown quantity: " + quantity)
        if quantity not in ["file", "folder"]:
            for f in derived_quantities[quantity]:
                if "field" not in f.keys():
                    raise KeyError("Missing key 'field'")
                if "surfaces" not in f.keys() and "volumes" not in f.keys():
                    raise KeyError("Missing key 'surfaces' or 'volumes'")

                field = f["field"]
                unknown_field = False
                if type(field) is str:
                    if field.isdigit():
                        field = int(field)
                    elif field not in FESTIM.helpers.field_types:
                        unknown_field = True

                if type(field) is int:
                    if field not in range(len(parameters["traps"]) + 1):
                        unknown_field = True

                if unknown_field:
                    raise ValueError("Unknown field: " + str(field))
