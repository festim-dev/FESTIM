def find_material_from_id(materials, id):
    ''' Returns the material from a given id
    Parameters:
    - materials : list of dicts
    - id : int
    '''
    for material in materials:
        if material['id'] == id:
            return material
            break
    print("Couldn't find ID " + str(id) + " in materials list")
    return


def update_expressions(expressions, t):
    '''
    Arguments:
    - expressions: list, contains the fenics Expression
    to be updated.
    - t: float, time.
    Update all FEniCS Expression() in expressions.
    '''
    for expression in expressions:
        expression.t = t
    return expressions


bc_types = {
    "dc": ["dc", "solubility", "table"],
    "neumann": ["flux"],
    "robin": ["recomb"]
}

quantity_types = [
    "surface_flux",
    "average_volume",
    "average_surface",
    "maximum_volume",
    "minimum_volume",
    "total_volume",
    "total_surface"
    ]

field_types = [
    "solute",
    "retention",
    "T"
]
parameters = {
    "materials": {
        "E_D": {
            "description": "Diffusion coefficient activation energy",
            "unit": "eV"
            },
        "D_0": {
            "description": "Diffusion coefficient pre exponential factor",
            "unit": "(m^2/s)"
            },
        "S_0": {
            "description": "Solubility coefficient pre exponential factor",
            "unit": "(m^2/s)"
            },
        "E_S": {
            "description": "Solubility coefficient activation energy",
            "unit": "eV"
            },
        "borders": {
            "description": "1D only: delimitations of the domain. \
            Exemple: [0, 0.5]",
            "unit": "m"
            },
        "thermal_cond": {
            "description": "required if heat equation is solved or if thermal\
                 flux is computed. thermal conductivity",
            "unit": "W/m/K"
            },
        "heat_capacity": {
            "description": "required if heat equation is solved. \
                Heat capacity",
            "unit": "J/K/kg"
            },
        "rho": {
            "description": "required if heat equation is solved. \
                Density",
            "unit": "kg/m^3"
            },
        "id": {
            "description": "id of the domain. If a marked mesh is given,\
                id must correspond to the volume_markers_file",
            "unit": "None"
            },
    },
    "traps": {
        "E_k": {
            "description": "Trapping rate activation energy",
            "unit": "eV"
            },
        "k_0": {
            "description": "Trapping rate pre-exponential factor",
            "unit": "m^3/s"
            },
        "E_p": {
            "description": "Detrapping rate activation energy",
            "unit": "eV"
            },
        "p_0": {
            "description": "Detrapping rate pre-exponential factor",
            "unit": "s^-1"
            },
        "density": {
            "description": "denisty of the trap. Can be float or an expression\
                 (ex: (1 + FESTIM.x)*FESTIM.t<100)",
            "unit": "m^-3"
            },
        "materials": {
            "description": "ids of the domains where the trap is present.\
                 Can be int or list of int",
            "unit": "None"
            },
        "source_term": {
            "description": "Volumetric source term for the trapped population",
            "unit": "m^-3.s^-1"
            },
        },
    }


def help_key(key):
    def find_dict(key, var, res=[]):
        if key in var:
            res.append(var[key])
        else:
            for k in var.keys():
                if type(var[k]) == dict:
                    find_dict(key, var=var[k], res=res)
        return res
    d = find_dict(key, parameters)
    for k in d:
        print(k)
