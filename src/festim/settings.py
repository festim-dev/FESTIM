from typing import Literal

import festim as F


class Settings:
    """Settings for a festim simulation.

    Args:
        atol (float): Absolute tolerance for the solver.
        rtol (float): Relative tolerance for the solver.
        max_iterations (int, optional): Maximum number of iterations for the
            solver. Defaults to 30.
        transient (bool, optional): Whether the simulation is transient or not.
        final_time (float, optional): Final time for a transient simulation.
            Defaults to None
        stepsize (festim.Stepsize, optional): stepsize for a transient
            simulation. Defaults to None
        convergence_criterion: resiudal or incremental (for Newton solver)

    Attributes:
        atol (float): Absolute tolerance for the solver.
        rtol (float): Relative tolerance for the solver.
        max_iterations (int): Maximum number of iterations for the solver.
        transient (bool): Whether the simulation is transient or not.
        final_time (float): Final time for a transient simulation.
        stepsize (festim.Stepsize): stepsize for a transient
            simulation.
        convergence_criterion: resiudal or incremental (for Newton solver)

    """

    def __init__(
        self,
        atol,
        rtol,
        max_iterations=30,
        transient=True,
        final_time=None,
        stepsize=None,
        convergence_criterion: Literal["residual", "incremental"] = "residual",
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.max_iterations = max_iterations
        self.transient = transient
        self.final_time = final_time
        self.stepsize = stepsize
        self.convergence_criterion = convergence_criterion

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
