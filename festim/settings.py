class Settings:
    """Settings for a festim simulation.

    Args:
        absolute_tolerance (float): Absolute tolerance for the solver.
        relative_tolerance (float): Relative tolerance for the solver.
        max_iterations (int, optional): Maximum number of iterations for the solver.
        final_time (float, optional): Final time for a transient simulation.
        stepsize (festim.Stepsize, optional): stepsize for a transient
            simulation.

    Attributes:
        aboslute_tolerance (float): Absolute tolerance for the solver.
        relative_tolerance (float): Relative tolerance for the solver.
        max_iterations (int, optional): Maximum number of iterations for the solver.
        final_time (float, optional): Final time for a transient simulation.
        stepsize (festim.Stepsize, optional): stepsize for a transient
            simulation.
    """

    def __init__(
        self,
        absolute_tolerance,
        relative_tolerance,
        max_iterations=30,
        final_time=None,
        stepsize=None,
    ) -> None:
        self.aboslute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.max_iterations = max_iterations
        self.final_time = final_time
        self.stepsize = stepsize
