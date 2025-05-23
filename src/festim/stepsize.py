import numpy as np


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
        max_stepsize (float or callable, optional): Maximum stepsize.
            If callable, has to be a function of `t`. Defaults to None.
        milestones (list, optional): list of times by which the simulation must
            pass. Defaults to an empty list.


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
        max_stepsize (float, callable): Maximum stepsize.
        milestones (list): list of times by which the simulation must
            pass.
    """

    def __init__(
        self,
        initial_value,
        growth_factor=None,
        cutback_factor=None,
        target_nb_iterations=None,
        max_stepsize=None,
        milestones=None,
    ) -> None:
        self.initial_value = initial_value
        self.growth_factor = growth_factor
        self.cutback_factor = cutback_factor
        self.target_nb_iterations = target_nb_iterations
        self.max_stepsize = max_stepsize
        self.milestones = milestones or []

        # TODO should this class hold the dt object used in the formulation

    @property
    def milestones(self):
        return self._milestones

    @milestones.setter
    def milestones(self, value):
        if value:
            self._milestones = sorted(value)
        else:
            self._milestones = value

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

    @property
    def max_stepsize(self):
        return self._max_stepsize

    @max_stepsize.setter
    def max_stepsize(self, value):
        if isinstance(value, float):
            if value < self.initial_value:
                raise ValueError(
                    "maximum stepsize cannot be less than initial stepsize"
                )

        self._max_stepsize = value

    def get_max_stepsize(self, t):
        """
        Returns the maximum stepsize at time t.

        Args:
            t (float): the current time

        Returns:
            float or None: the maximum stepsize at time t
        """
        if callable(self._max_stepsize):
            return self._max_stepsize(t)
        return self._max_stepsize

    def modify_value(self, value, nb_iterations, t=None):
        if not self.is_adapt(t):
            return value

        if nb_iterations < self.target_nb_iterations:
            updated_value = value * self.growth_factor
        elif nb_iterations > self.target_nb_iterations:
            updated_value = value * self.cutback_factor
        else:
            updated_value = value

        if max_step := self.get_max_stepsize(t):
            if updated_value > max_step:
                updated_value = max_step

        next_milestone = self.next_milestone(t)
        if next_milestone is not None:
            time_to_milestone = next_milestone - t
            if updated_value > time_to_milestone and not np.isclose(
                t, next_milestone, atol=0
            ):
                updated_value = time_to_milestone

        return updated_value

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

    def next_milestone(self, current_time: float):
        """Returns the next milestone that the simulation must pass.
        Returns None if there are no more milestones.

        Args:
            current_time (float): current time.

        Returns:
            float: next milestone.
        """
        if self.milestones is None:
            return None
        for milestone in self.milestones:
            if current_time < milestone:
                return milestone
        return None
