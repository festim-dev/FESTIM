import FESTIM
import fenics as f
import sympy as sp


class BoundaryCondition:
    def __init__(self, surfaces, component=0) -> None:

        if not isinstance(surfaces, list):
            surfaces = [surfaces]
        self.surfaces = surfaces

        self.component = component
        self.expression = None
        self.sub_expressions = []
