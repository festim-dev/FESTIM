import fenics as f
import numpy as np
import warnings


class Stepsize:
    """
    Description of Stepsize

    Args:
        initial_value (float, optional): initial stepsize. Defaults to 0.0.
        stepsize_change_ratio (float, optional): stepsize change ratio.
            Defaults to None.
        t_stop (float, optional): time at which the adaptive stepsize
            stops. Defaults to None.
        stepsize_stop_max (float, optional): Maximum stepsize after
            t_stop. Defaults to None.
        max_stepsize (float or callable, optional): Maximum stepsize.
            Can be a function of festim.t. Defaults to None.
        dt_min (float, optional): Minimum stepsize below which an error is
            raised. Defaults to None.
        milestones (list, optional): list of times by which the simulation must
            pass. Defaults to None.

    Attributes:
        adaptive_stepsize (dict): contains the parameters for adaptive stepsize
        value (fenics.Constant): value of dt
        milestones (list): list of times by which the simulation must
            pass.

    Example::

        my_stepsize = Stepsize(
            initial_value=0.5,
            stepsize_change_ratio=1.1,
            max_stepsize=lambda t: None if t < 1 else 2,
            dt_min=1e-05
        )
    """

    def __init__(
        self,
        initial_value=0.0,
        stepsize_change_ratio=None,
        t_stop=None,
        stepsize_stop_max=None,
        max_stepsize=None,
        dt_min=None,
        milestones=None,
    ) -> None:
        self.adaptive_stepsize = None
        if stepsize_change_ratio is not None:
            if t_stop or stepsize_stop_max:
                warnings.warn(
                    "stepsize_stop_max and t_stop attributes will be deprecated in a future release, please use max_stepsize instead",
                    DeprecationWarning,
                )
                max_stepsize = lambda t: stepsize_stop_max if t >= t_stop else None
            self.adaptive_stepsize = {
                "stepsize_change_ratio": stepsize_change_ratio,
                "max_stepsize": max_stepsize,
                "dt_min": dt_min,
            }
        self.initial_value = initial_value
        self.value = None
        self.milestones = milestones
        self.initialise_value()

    @property
    def milestones(self):
        return self._milestones

    @milestones.setter
    def milestones(self, value):
        if value:
            self._milestones = sorted(value)
        else:
            self._milestones = value

    def initialise_value(self):
        """Creates a fenics.Constant object initialised with self.initial_value
        and stores it in self.value"""
        self.value = f.Constant(self.initial_value, name="dt")

    def adapt(self, t, nb_it, converged):
        """Changes the stepsize based on convergence.

        Args:
            t (float): current time.
            nb_it (int): number of iterations the solver required to converge.
            converged (bool): True if the solver converged, else False.
        """
        if self.adaptive_stepsize:
            change_ratio = self.adaptive_stepsize["stepsize_change_ratio"]
            dt_min = self.adaptive_stepsize["dt_min"]
            max_stepsize = self.adaptive_stepsize["max_stepsize"]

            if not converged:
                self.value.assign(float(self.value) / change_ratio)
                if float(self.value) < dt_min:
                    raise ValueError("stepsize reached minimal value")
            if nb_it < 5:
                self.value.assign(float(self.value) * change_ratio)
            else:
                self.value.assign(float(self.value) / change_ratio)

            if callable(max_stepsize):
                max_stepsize = max_stepsize(t)
            if max_stepsize is not None:
                if float(self.value) > max_stepsize:
                    self.value.assign(max_stepsize)

        # adapt for next milestone
        next_milestone = self.next_milestone(t)
        if next_milestone is not None:
            if t + float(self.value) > next_milestone and not np.isclose(
                t, next_milestone, atol=0
            ):
                self.value.assign((next_milestone - t))

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
