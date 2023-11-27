import festim as F


class Stepsize:
    """
    A class for evaluating the stepsize of transient simulations.

    Args:
        initial_value (float, int): initial stepsize.

    Attributes:
        initial_value (float, int): initial stepsize.
    """

    def __init__(
        self,
        initial_value,
    ) -> None:
        self.initial_value = initial_value
        self.growth_factor = 1.2
        self.cutback_factor = 0.8
        self.target_nb_iterations = 4

    def adapt(self, value, nb_iterations):
        new_value = value
        if nb_iterations < self.target_nb_iterations:
            new_value = value * self.growth_factor
        elif nb_iterations > self.target_nb_iterations:
            new_value = value * self.cutback_factor

        return new_value
