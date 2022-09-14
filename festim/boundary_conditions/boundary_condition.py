class BoundaryCondition:
    """Base BoundaryCondition class

    Args:
        surfaces (list or int): the surfaces of the BC
        field (int or str): the field the boundary condition is
            applied to. 0 and "solute" stand for the mobile
            concentration, "T" for temperature
    """

    def __init__(self, surfaces, field) -> None:

        if not isinstance(surfaces, list):
            surfaces = [surfaces]
        self.surfaces = surfaces

        if field == "solute":
            self.field = 0
        else:
            self.field = field
        self.expression = None
        self.sub_expressions = []
