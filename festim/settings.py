class Settings:
    """Settings for a festim simulation.

    Args:
        atol (float): Absolute tolerance for the solver.
        rtol (float): Relative tolerance for the solver.
        max_iterations (int, optional): Maximum number of iterations for the solver.
        final_time (float, optional): Final time for a transient simulation.
        stepsize (festim.Stepsize, optional): stepsize for a transient
            simulation.

    Attributes:
        atol (float): Absolute tolerance for the solver.
        rtol (float): Relative tolerance for the solver.
        max_iterations (int, optional): Maximum number of iterations for the solver.
        final_time (float, optional): Final time for a transient simulation.
        stepsize (festim.Stepsize, optional): stepsize for a transient
            simulation.
    """

    def __init__(
        self,
        atol,
        rtol,
        max_iterations=30,
        final_time=None,
        stepsize=None,
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.max_iterations = max_iterations
        self.final_time = final_time
        self.stepsize = stepsize
