from FESTIM import DirichletBC, BoundaryConditionExpression


class CustomDirichlet(DirichletBC):
    def __init__(self, surfaces, function, component=0, **prms) -> None:
        super().__init__(surfaces, component=component, **prms)
        self.function = function

    def create_expression(self, T):
        value_BC = BoundaryConditionExpression(
            T, self.function,
            **self.prms,
        )
        self.expression = value_BC
        self.sub_expressions = self.prms.values()
