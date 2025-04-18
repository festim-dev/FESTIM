class SurfaceSubdomain:
    """
    Surface subdomain class

    Args:
        id (int): the id of the surface subdomain
    """

    id: int

    def __init__(self, id):
        self.id = id


def find_surface_from_id(id: int, surfaces: list):
    """Returns the correct surface subdomain object from a list of surface ids
    based on an int

    Args:
        id (int): the id of the surface subdomain
        surfaces (list of F.SurfaceSubdomain): the list of surfaces

    Returns:
        festim.SurfaceSubdomain: the surface subdomain object with the correct id

    Raises:
        ValueError: if the surface name is not found in the list of surfaces

    """
    for surf in surfaces:
        if surf.id == id:
            return surf
    raise ValueError(f"id {id} not found in list of surfaces")
