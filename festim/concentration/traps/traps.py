import festim
import fenics as f
import warnings


class Traps(list):
    """
    A list of festim.Trap objects
    """

    def __init__(self, *args):
        # checks that input is list
        if len(args) == 0:
            super().__init__()
        else:
            if not isinstance(*args, list):
                raise TypeError("festim.Traps must be a list")
            super().__init__(self._validate_trap(item) for item in args[0])

        self.F = None
        self.extrinsic_formulations = []
        self.sub_expressions = []

    @property
    def traps(self):
        warnings.warn(
            "The traps attribute will be deprecated in a future release, please use festim.Traps as a list instead",
            DeprecationWarning,
        )
        return self

    @traps.setter
    def traps(self, value):
        warnings.warn(
            "The traps attribute will be deprecated in a future release, please use festim.Traps as a list instead",
            DeprecationWarning,
        )
        if isinstance(value, list):
            if not all(isinstance(t, festim.Trap) for t in value):
                raise TypeError("traps must be a list of festim.Trap")
            super().__init__(value)
        else:
            raise TypeError("traps must be a list")

    def __setitem__(self, index, item):
        super().__setitem__(index, self._validate_trap(item))

    def insert(self, index, item):
        super().insert(index, self._validate_trap(item))

    def append(self, item):
        super().append(self._validate_trap(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(self._validate_trap(item) for item in other)

    def _validate_trap(self, value):
        if isinstance(value, festim.Trap):
            return value
        raise TypeError("festim.Traps must be a list of festim.Trap")

    def make_traps_materials(self, materials):
        for trap in self:
            trap.make_materials(materials)

    def assign_traps_ids(self):
        for i, trap in enumerate(self, 1):
            if trap.id is None:
                trap.id = i

    def create_forms(self, mobile, materials, T, dx, dt=None):
        self.F = 0
        for trap in self:
            trap.create_form(mobile, materials, T, dx, dt=dt)
            self.F += trap.F
            self.sub_expressions += trap.sub_expressions

    def get_trap(self, id):
        for trap in self:
            if trap.id == id:
                return trap
        raise ValueError("Couldn't find trap {}".format(id))

    def initialise_extrinsic_traps(self, V):
        """Add functions to ExtrinsicTrapBase objects for density form"""
        for trap in self:
            if isinstance(trap, festim.ExtrinsicTrapBase):
                trap.density = [f.Function(V)]
                trap.density_test_function = f.TestFunction(V)
                trap.density_previous_solution = f.project(f.Constant(0), V)

    def define_variational_problem_extrinsic_traps(self, dx, dt, T):
        """
        Creates the variational formulations for the extrinsic traps densities

        Args:
            dx (fenics.Measure): the dx measure of the sim
            dt (festim.Stepsize): If None assuming steady state.
            T (festim.Temperature): the temperature of the simulation
        """
        self.extrinsic_formulations = []
        expressions_extrinsic = []
        for trap in self:
            if isinstance(trap, festim.ExtrinsicTrapBase):
                trap.create_form_density(dx, dt, T)
                self.extrinsic_formulations.append(trap.form_density)
        self.sub_expressions.extend(expressions_extrinsic)

    def define_newton_solver_extrinsic_traps(self):
        for trap in self:
            if isinstance(trap, festim.ExtrinsicTrapBase):
                trap.define_newton_solver()

    def solve_extrinsic_traps(self):
        for trap in self:
            if isinstance(trap, festim.ExtrinsicTrapBase):
                du_t = f.TrialFunction(trap.density[0].function_space())
                J_t = f.derivative(trap.form_density, trap.density[0], du_t)
                problem = festim.Problem(J_t, trap.form_density, [])

                f.begin(
                    "Solving nonlinear variational problem."
                )  # Add message to fenics logs
                trap.newton_solver.solve(problem, trap.density[0].vector())
                f.end()

    def update_extrinsic_traps_density(self):
        for trap in self:
            if isinstance(trap, festim.ExtrinsicTrapBase):
                trap.density_previous_solution.assign(trap.density[0])
