from typing import Literal

import festim as F


class Settings:
    """Settings for a festim simulation.

    Args:
        atol (float or callable): Absolute tolerance for the solver.
        rtol (float or callable): Relative tolerance for the solver.
        max_iterations (int, optional): Maximum number of iterations for the
            solver. Defaults to 30.
        transient (bool, optional): Whether the simulation is transient or not.
        final_time (float, optional): Final time for a transient simulation.
            Defaults to None
        element_degree (int, optional): Degree order for finite element.
            Defaults to 1.
        stepsize (festim.Stepsize, optional): stepsize for a transient
            simulation. Defaults to None
        convergence_criterion: resiudal or incremental (for Newton solver)
        export_time_atol (float, optional): Absolute tolerance used to decide
            whether the current time matches one of the ``times`` requested by
            an export. Defaults to 0.
        export_time_rtol (float, optional): Relative tolerance (with respect to
            the current time) used to decide whether the current time matches
            one of the ``times`` requested by an export. At large times the
            relative tolerance alone can be very wide, and ``export_time_atol``
            should be used instead. Defaults to 1e-5.

    Attributes:
        atol (float or callable): Absolute tolerance for the solver.
        rtol (float or callable): Relative tolerance for the solver.
        max_iterations (int): Maximum number of iterations for the solver.
        transient (bool): Whether the simulation is transient or not.
        final_time (float): Final time for a transient simulation.
        element_degree (int): Degree order for finite element.
        stepsize (festim.Stepsize): stepsize for a transient
            simulation.
        convergence_criterion: resiudal or incremental (for Newton solver)
        export_time_atol (float): Absolute tolerance for matching export times.
        export_time_rtol (float): Relative tolerance for matching export times.
    """

    def __init__(
        self,
        atol,
        rtol,
        max_iterations=30,
        transient=True,
        final_time=None,
        element_degree=1,
        stepsize=None,
        convergence_criterion: Literal["residual", "incremental"] = "residual",
        export_time_atol: float = 0.0,
        export_time_rtol: float = 1e-5,
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.max_iterations = max_iterations
        self.transient = transient
        self.final_time = final_time
        self.element_degree = element_degree
        self.stepsize = stepsize
        self.convergence_criterion = convergence_criterion
        self.export_time_atol = export_time_atol
        self.export_time_rtol = export_time_rtol

    @property
    def export_time_atol(self):
        return self._export_time_atol

    @export_time_atol.setter
    def export_time_atol(self, value):
        if value < 0:
            raise ValueError("export_time_atol should be greater than or equal to zero")
        self._export_time_atol = value

    @property
    def export_time_rtol(self):
        return self._export_time_rtol

    @export_time_rtol.setter
    def export_time_rtol(self, value):
        if value < 0:
            raise ValueError("export_time_rtol should be greater than or equal to zero")
        self._export_time_rtol = value

    @property
    def stepsize(self):
        return self._stepsize

    @stepsize.setter
    def stepsize(self, value):
        if value is None:
            self._stepsize = None
        elif isinstance(value, (float, int)):
            self._stepsize = F.Stepsize(initial_value=value)
        elif isinstance(value, F.Stepsize):
            self._stepsize = value
        else:
            raise TypeError("stepsize must be an of type int, float or festim.Stepsize")
