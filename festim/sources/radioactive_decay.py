from festim import Source


class RadioactiveDecay(Source):
    def __init__(self, decay_constant, volume, field="all") -> None:
        self.decay_constant = decay_constant
        super().__init__(value=None, volume=volume, field=field)

    def form(self, concentration):
        return self.decay_constant * concentration
