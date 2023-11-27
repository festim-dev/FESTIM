import festim as F


class Stepsize:
    """
    A class for evaluating the stepsize of transient simulations.

    Args:
        initial_value (float, int): initial stepsize.

    Attributes:
        initial_value (float, int): initial stepsize.
        growth_factor (float): factor by which the stepsize is
            increased when adapting
        cutback_factor (float): factor by which the stepsize is
            decreased when adapting
        target_nb_iterations (int): number of Newton iterations
            over (resp. under) which the stepsize is increased
            (resp. decreased)
    """

    def __init__(
        self,
        initial_value,
    ) -> None:
        self.initial_value = initial_value
        self.growth_factor = 1.2
        self.cutback_factor = 0.8
        self.target_nb_iterations = 4

        # TODO should this class hold the dt object used in the formulation

    def modify_value(self, value, nb_iterations, t=None):
        if not self.is_adapt(t):
            return value

        if nb_iterations < self.target_nb_iterations:
            return value * self.growth_factor
        elif nb_iterations > self.target_nb_iterations:
            return value * self.cutback_factor
        else:
            return value

    def is_adapt(self, t):
        """
        Methods that defines if the stepsize should be
        adapted or not

        Args:
            t (float): the current time

        Returns:
            bool: True if needs to adapt, False otherwise.
        """
        return True
