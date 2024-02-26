from festim import Source


class RadioactiveDecay(Source):
    """
    A radioactive decay source.
    dc/dt = ... - lambda * c
    where lambda is the decay constant in 1/s.

    Args:
        decay_constant (float): The decay constant of the source in 1/s.
        volume (int): The volume of the source.
        field (str, optional): The field to which the source is
            applied. If "all" the decay will be applied to all
            concentrations. Defaults to "all".
    """

    def __init__(self, decay_constant, volume, field="all") -> None:
        self.decay_constant = decay_constant
        super().__init__(value=None, volume=volume, field=field)

    @property
    def decay_constant(self):
        return self._decay_constant

    @decay_constant.setter
    def decay_constant(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError("decay_constant must be a float or an int")
        if value <= 0:
            raise ValueError("decay_constant must be positive")
        self._decay_constant = value

    def form(self, concentration):
        return -self.decay_constant * concentration
