from fenics import *
import sympy as sp
import numpy as np
import FESTIM


def compute_error(parameters, t, u_n, mesh):
    '''
    Returns a list containing the errors
    '''
    res = u_n.split()
    tab = []
    for error in parameters:
        er = []
        er.append(t)
        for i in range(len(error["exact_solution"])):
            sol = Expression(sp.printing.ccode(error["exact_solution"][i]),
                             degree=error["degree"], t=t)
            if error["norm"] == "error_max":
                vertex_values_u = res[i].compute_vertex_values(mesh)
                vertex_values_sol = sol.compute_vertex_values(mesh)
                error_max = np.max(np.abs(vertex_values_u - vertex_values_sol))
                er.append(error_max)
            else:
                error_L2 = errornorm(
                    sol, res[i], error["norm"])
                er.append(error_L2)

        tab.append(er)
    return tab


def compute_retention(u, W):
    res = list(split(u))
    if not res:  # if u is non-vector
        res = [u]
    retention = project(res[0])
    total_trap = 0
    for i in range(1, len(res)):
        sol = res[i]
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

    # Update coefficient D_0 and E_diff
    for cell in cells(mesh):

        subdomain_id = volume_markers[cell]
        material = FESTIM.helpers.find_material_from_id(
            materials, subdomain_id)
        value_D0 = material["D_0"]
        value_E_diff = material["E_diff"]
        if "thermal_cond" in material:
            value_thermal_cond = material["thermal_cond"]
        cell_no = cell.index()
        D_0.vector()[cell_no] = value_D0
        E_diff.vector()[cell_no] = value_E_diff
        thermal_cond.vector()[cell_no] = value_thermal_cond
    return D_0, E_diff, thermal_cond


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
    header = ['t(s)']
    i = 0
    for flux in parameters["exports"]["derived_quantities"]["surface_flux"]:
        for surf in flux["surfaces"]:
            header.append("Flux surface " + str(surf) + ": " + flux['field'])
    for average in parameters[
                    "exports"]["derived_quantities"]["average_volume"]:
        for vol in average["volumes"]:
            header.append(
                "Average " + average['field'] + " volume " + str(vol))
    for minimum in parameters[
                    "exports"]["derived_quantities"]["minimum_volume"]:
        for vol in minimum["volumes"]:
            header.append(
                "Minimum " + minimum["field"] + " volume " + str(vol))
    for maximum in parameters[
                    "exports"]["derived_quantities"]["maximum_volume"]:
        for vol in maximum["volumes"]:
            header.append(
                "Maximum " + maximum["field"] + " volume " + str(vol))
    for total in parameters["exports"]["derived_quantities"]["total_volume"]:
        for vol in total["volumes"]:
            header.append("Total " + total["field"] + " volume " + str(vol))
    for total in parameters["exports"]["derived_quantities"]["total_surface"]:
        for surf in total["surfaces"]:
            header.append("Total " + total["field"] + " surface " + str(surf))

    return header


def derived_quantities(parameters, solutions, properties, markers):
    '''
    Computes all the derived_quantities and store it into list
    '''
    D = properties[0]
    thermal_cond = properties[1]

    volume_markers = markers[0]
    surface_markers = markers[1]
    V = solutions[0].function_space()
    mesh = V.mesh()
    n = FacetNormal(mesh)
    dx = Measure('dx', domain=mesh, subdomain_data=volume_markers)
    ds = Measure('ds', domain=mesh, subdomain_data=surface_markers)

    # Create dicts
    field_to_sol = {
        'solute': solutions[0],
        'retention': solutions[len(solutions)-2],
        'T': solutions[len(solutions)-1],
    }
    field_to_prop = {
        'solute': D,
        'T': thermal_cond,
    }
    for i in range(1, len(solutions)-2):
        field_to_sol[str(i)] = solutions[i]

    for key, val in field_to_sol.items():
        if isinstance(val, function.expression.Expression):
            val = interpolate(val, V)
            field_to_sol[key] = val
    tab = []
    # Compute quantities
    for flux in parameters["exports"]["derived_quantities"]["surface_flux"]:
        sol = field_to_sol[flux["field"]]
        prop = field_to_prop[flux["field"]]
        for surf in flux["surfaces"]:
            tab.append(assemble(prop*dot(grad(sol), n)*ds(surf)))
    for average in parameters[
                    "exports"]["derived_quantities"]["average_volume"]:
        sol = field_to_sol[average["field"]]
        for vol in average["volumes"]:
            val = assemble(sol*dx(vol))/assemble(1*dx(vol))
            tab.append(val)
    for minimum in parameters[
                    "exports"]["derived_quantities"]["minimum_volume"]:
        sol = field_to_sol[minimum["field"]]
        for vol in minimum["volumes"]:
            tab.append(calculate_minimum_volume(sol, volume_markers, vol))
    for maximum in parameters[
                    "exports"]["derived_quantities"]["maximum_volume"]:
        sol = field_to_sol[maximum["field"]]
        for vol in maximum["volumes"]:
            tab.append(calculate_maximum_volume(sol, volume_markers, vol))
    for total in parameters["exports"]["derived_quantities"]["total_volume"]:
        sol = field_to_sol[total["field"]]
        for vol in total["volumes"]:
            tab.append(assemble(sol*dx(vol)))
    for total in parameters["exports"]["derived_quantities"]["total_surface"]:
        sol = field_to_sol[total["field"]]
        for surf in total["surfaces"]:
            tab.append(assemble(sol*ds(surf)))
    return tab
