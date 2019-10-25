from fenics import *
import sympy as sp
import numpy as np
import FESTIM


def run_post_processing(parameters, transient, u, T, markers, W, t, dt, files,
                        append, flux_fonctions, derived_quantities_global):
    if u.function_space().num_sub_spaces() == 0:
        res = [u]
    else:
        res = list(u.split())
    retention = FESTIM.post_processing.compute_retention(u, W)
    res.append(retention)
    if isinstance(T, function.expression.Expression):
        res.append(interpolate(T, W))
    else:
        res.append(T)

    if "derived_quantities" in parameters["exports"].keys():
        D_0, E_diff, thermal_cond, G, S = flux_fonctions
        derived_quantities_t = \
            FESTIM.post_processing.derived_quantities(
                parameters,
                res,
                [D_0*exp(-E_diff/T), thermal_cond],
                markers)
        derived_quantities_t.insert(0, t)
        derived_quantities_global.append(derived_quantities_t)
    if "xdmf" in parameters["exports"].keys():
        FESTIM.export.export_xdmf(
            res, parameters["exports"], files, t, append=append)
    if "txt" in parameters["exports"].keys():
        dt = FESTIM.export.export_profiles(res, parameters["exports"], t, dt, W)

    return derived_quantities_global, dt


def compute_error(parameters, t, res, mesh):
    '''
    Returns a list containing the errors
    '''
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
            try:
                nb = int(error["computed_solutions"][i])
                computed_sol = res[nb]
            except:
                try:
                    computed_sol = solution_dict[
                        error["computed_solutions"][i]]
                except:
                    raise KeyError(
                        "The key " + error["computed_solutions"][i] +
                        " is unknown.")

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


def compute_retention(u, W):
    res = list(split(u))
    if not res:  # if u is non-vector
        res = [u]
    retention = project(res[0])
    for i in range(1, len(res)):
        retention = project(retention + res[i], W)
    return retention


def create_flux_functions(mesh, materials, volume_markers):
    '''
    Returns Function() objects for fluxes computation
    '''
    D0 = FunctionSpace(mesh, 'DG', 0)
    D_0 = Function(D0, name="D_0")
    E_diff = Function(D0, name="E_diff")
    thermal_cond = Function(D0, name="thermal_cond")
    G = Function(D0, name="G")
    S = Function(D0, name="S")

    # Update coefficient D_0 and E_diff
    for cell in cells(mesh):

        subdomain_id = volume_markers[cell]
        material = FESTIM.helpers.find_material_from_id(
            materials, subdomain_id)
        value_D0 = material["D_0"]
        value_E_diff = material["E_diff"]
        cell_no = cell.index()
        if "thermal_cond" in material:
            thermal_cond.vector()[cell_no] = material["thermal_cond"]
        if "H" in material:
            G.vector()[cell_no] = material["H"]["free_enthalpy"]
            S.vector()[cell_no] = material["H"]["entropy"]
        D_0.vector()[cell_no] = value_D0
        E_diff.vector()[cell_no] = value_E_diff
    return D_0, E_diff, thermal_cond, G, S


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

    header = ['t(s)']
    if "surface_flux" in parameters["exports"]["derived_quantities"].keys():
        for flux in parameters["exports"]["derived_quantities"]["surface_flux"]:
            for surf in flux["surfaces"]:
                header.append("Flux surface " + str(surf) + ": " + str(flux['field']))
    if "average_volume" in parameters["exports"]["derived_quantities"].keys():
        for average in parameters[
                        "exports"]["derived_quantities"]["average_volume"]:
            for vol in average["volumes"]:
                header.append(
                    "Average " + str(average['field']) + " volume " + str(vol))
    if "minimum_volume" in parameters["exports"]["derived_quantities"].keys():
        for minimum in parameters[
                        "exports"]["derived_quantities"]["minimum_volume"]:
            for vol in minimum["volumes"]:
                header.append(
                    "Minimum " + str(minimum["field"]) + " volume " + str(vol))
    if "maximum_volume" in parameters["exports"]["derived_quantities"].keys():
        for maximum in parameters[
                        "exports"]["derived_quantities"]["maximum_volume"]:
            for vol in maximum["volumes"]:
                header.append(
                    "Maximum " + str(maximum["field"]) + " volume " + str(vol))
    if "total_volume" in parameters["exports"]["derived_quantities"].keys():
        for total in parameters["exports"]["derived_quantities"]["total_volume"]:
            for vol in total["volumes"]:
                header.append("Total " + str(total["field"]) + " volume " + str(vol))
    if "total_surface" in parameters["exports"]["derived_quantities"].keys():
        for total in parameters["exports"]["derived_quantities"]["total_surface"]:
            for surf in total["surfaces"]:
                header.append("Total " + str(total["field"]) + " surface " + str(surf))

    return header


def derived_quantities(parameters, solutions, properties, markers):
    '''
    Computes all the derived_quantities and store it into list
    '''
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
    V = solutions[0].function_space()
    mesh = V.mesh()
    W = FunctionSpace(mesh, 'P', 1)
    n = FacetNormal(mesh)
    dx = Measure('dx', domain=mesh, subdomain_data=volume_markers)
    ds = Measure('ds', domain=mesh, subdomain_data=surface_markers)

    # Create dicts
    solute = solutions[0]
    ret = solutions[len(solutions)-2]
    T = solutions[len(solutions)-1]
    field_to_sol = {
        'solute': solute,
        'retention': ret,
        'T': T,
    }
    field_to_prop = {
        'solute': D,
        'T': thermal_cond,
    }
    for i in range(1, len(solutions)-2):
        field_to_sol[str(i)] = solutions[i]

    for key, val in field_to_sol.items():
        if isinstance(val, function.expression.Expression):
            val = interpolate(val, W)
            field_to_sol[key] = val
    tab = []
    # Compute quantities
    if "surface_flux" in parameters["exports"]["derived_quantities"].keys():
        for flux in parameters["exports"]["derived_quantities"]["surface_flux"]:
            sol = field_to_sol[str(flux["field"])]
            prop = field_to_prop[str(flux["field"])]
            for surf in flux["surfaces"]:
                phi = assemble(prop*dot(grad(sol), n)*ds(surf))
                if soret is True and str(flux["field"]) == 'solute':
                    phi += assemble(
                        prop*sol*Q/(FESTIM.R*T**2)*dot(grad(T), n)*ds(surf))
                tab.append(phi)
    if "average_volume" in parameters["exports"]["derived_quantities"].keys():
        for average in parameters[
                        "exports"]["derived_quantities"]["average_volume"]:
            sol = field_to_sol[str(average["field"])]
            for vol in average["volumes"]:
                val = assemble(sol*dx(vol))/assemble(1*dx(vol))
                tab.append(val)
    if "minimum_volume" in parameters["exports"]["derived_quantities"].keys():
        for minimum in parameters[
                        "exports"]["derived_quantities"]["minimum_volume"]:
            sol = field_to_sol[str(minimum["field"])]
            for vol in minimum["volumes"]:
                tab.append(calculate_minimum_volume(sol, volume_markers, vol))
    if "maximum_volume" in parameters["exports"]["derived_quantities"].keys():
        for maximum in parameters[
                        "exports"]["derived_quantities"]["maximum_volume"]:
            sol = field_to_sol[str(maximum["field"])]
            for vol in maximum["volumes"]:
                tab.append(calculate_maximum_volume(sol, volume_markers, vol))
    if "total_volume" in parameters["exports"]["derived_quantities"].keys():
        for total in parameters["exports"]["derived_quantities"]["total_volume"]:
            sol = field_to_sol[str(total["field"])]
            for vol in total["volumes"]:
                tab.append(assemble(sol*dx(vol)))
    if "total_surface" in parameters["exports"]["derived_quantities"].keys():
        for total in parameters["exports"]["derived_quantities"]["total_surface"]:
            sol = field_to_sol[str(total["field"])]
            for surf in total["surfaces"]:
                tab.append(assemble(sol*ds(surf)))
    return tab


def check_keys_derived_quantities(parameters):
    for quantity in parameters["exports"]["derived_quantities"].keys():
        if quantity not in [*FESTIM.helpers.quantity_types, "file", "folder"]:
            raise ValueError("Unknown quantity: " + quantity)
        if quantity not in ["file", "folder"]:
            for f in parameters["exports"]["derived_quantities"][quantity]:
                if "field" not in f.keys():
                    raise KeyError("Missing key 'field'")
                else:
                    if type(f["field"]) is int:
                        if f["field"] > len(parameters["traps"]) or f["field"] < 0:
                            raise ValueError("Unknown field: " + str(f["field"]))
                    elif type(f["field"]) is str:
                        if f["field"] not in FESTIM.helpers.field_types:
                            try:
                                if int(f["field"]) > len(parameters["traps"]) or int(f["field"])  < 0:
                                    raise ValueError("Unknown field: " + f["field"])
                            except:
                                raise ValueError("Unknown field: " + str(f["field"]))

                if "surfaces" not in f.keys() and "volumes" not in f.keys():
                    raise KeyError("Missing key 'surfaces' or 'volumes'")
