class Stepsize:
    """
    A class for evaluating the stepsize of transient simulations.

    Args:
        initial_value (float, int): initial stepsize.
        growth_factor (float, optional): factor by which the stepsize is
            increased when adapting
        cutback_factor (float, optional): factor by which the stepsize is
            decreased when adapting
        target_nb_iterations (int, optional): number of Newton iterations
            over (resp. under) which the stepsize is increased
            (resp. decreased)


    Attributes:
        initial_value (float, int): initial stepsize.
        growth_factor (float): factor by which the stepsize is
            increased when adapting
        cutback_factor (float): factor by which the stepsize is
            decreased when adapting
        target_nb_iterations (int): number of Newton iterations
            over (resp. under) which the stepsize is increased
            (resp. decreased)
        adaptive (bool): True if the stepsize is adaptive, False otherwise.
    """

    def __init__(
        self,
        initial_value,
        growth_factor=None,
        cutback_factor=None,
        target_nb_iterations=None,
    ) -> None:
        self.initial_value = initial_value
        self.growth_factor = growth_factor
        self.cutback_factor = cutback_factor
        self.target_nb_iterations = target_nb_iterations

        # TODO should this class hold the dt object used in the formulation

    @property
    def adaptive(self):
        return self.growth_factor or self.cutback_factor or self.target_nb_iterations

    @property
    def growth_factor(self):
        return self._growth_factor

    @growth_factor.setter
    def growth_factor(self, value):
        if value is not None:
            if value < 1:
                raise ValueError("growth factor should be greater than one")

        self._growth_factor = value

    @property
    def cutback_factor(self):
        return self._cutback_factor

    @cutback_factor.setter
    def cutback_factor(self, value):
        if value is not None:
            if value > 1:
                raise ValueError("cutback factor should be smaller than one")

        self._cutback_factor = value

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
