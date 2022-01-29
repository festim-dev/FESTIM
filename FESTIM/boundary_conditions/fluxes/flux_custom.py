from FESTIM import FluxBC


class CustomFlux(FluxBC):
    """FluxBC subclass allowing the use of a user-defined function.
    Usage:
    def fun(T, solute, param1):
        return 2*T + solute - param1
    my_bc = CustomFlux(surfaces=[1, 2], function=fun)
    """
    def __init__(self, surfaces, function, **kwargs) -> None:
        """Inits CustomFlux

        Args:
            surfaces (list or int): the surfaces of the BC
            function (callable): the function. Example:
                def fun(T, solute, param1):
                    return 2*T + solute - param1
        """
        super().__init__(surfaces=surfaces, component="T", **kwargs)
        self.function = function

    def create_form(self, T, solute):
        self.form = self.function(T, solute, self.prms)
        self.sub_expressions += [
            expression for expression in self.prms.values()]
