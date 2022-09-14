class BoundaryCondition:
    def __init__(self, surfaces, field) -> None:

        if not isinstance(surfaces, list):
            surfaces = [surfaces]
        self.surfaces = surfaces

        self.field = field
        self.expression = None
        self.sub_expressions = []
