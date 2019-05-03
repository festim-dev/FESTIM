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