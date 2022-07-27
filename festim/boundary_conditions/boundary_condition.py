import festim
import fenics as f
import sympy as sp


class BoundaryCondition:
    def __init__(self, surfaces, field=0) -> None:

        if not isinstance(surfaces, list):
            surfaces = [surfaces]
        self.surfaces = surfaces

        self.field = field
        self.expression = None
        self.sub_expressions = []
