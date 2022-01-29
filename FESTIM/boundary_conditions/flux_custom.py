from FESTIM import FluxBC


class CustomFlux(FluxBC):
    def __init__(self, surfaces, function, **kwargs) -> None:
        super().__init__(surfaces=surfaces, component="T", **kwargs)
        self.function = function

    def create_form(self, T, solute):
        self.form = self.function(T, solute, self.prms)
        self.sub_expressions += [
            expression for expression in self.prms.values()]
