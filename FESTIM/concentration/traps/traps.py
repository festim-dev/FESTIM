import FESTIM
import fenics as f


class Traps:
    def __init__(self, traps=[]) -> None:
        self.traps = traps
        self.F = None
        self.extrinsic_formulations = []
        self.sub_expressions = []

        # add ids if unspecified
        for i, trap in enumerate(self.traps, 1):
            if trap.id is None:
                trap.id = i

    def create_forms(self, mobile, materials, T, dx, dt=None,
                     chemical_pot=False):
        self.F = 0
        for trap in self.traps:
            trap.create_form(mobile, materials, T, dx, dt=dt,
                             chemical_pot=chemical_pot)
            self.F += trap.F
            self.sub_expressions += trap.sub_expressions

    def get_trap(self, id):
        for trap in self.traps:
            if trap.id == id:
                return trap
        raise ValueError("Couldn't find trap {}".format(id))

    def initialise_extrinsic_traps(self, V):
        """Add functions to ExtrinsicTrap objects for density form
        """
        for trap in self.traps:
            if isinstance(trap, FESTIM.ExtrinsicTrap):
                trap.density = [f.Function(V)]
                trap.density_test_function = f.TestFunction(V)
                trap.density_previous_solution = \
                    f.project(f.Constant(0), V)

    def define_variational_problem_extrinsic_traps(self, dx, dt, T):
        """
        Creates the variational formulations for the extrinsic traps densities

        Args:
            dx (fenics.Measure): the dx measure of the sim
            dt (FESTIM.Stepsize): If None assuming steady state.
            T (FESTIM.Temperature): the temperature of the simulation
        """
        self.extrinsic_formulations = []
        expressions_extrinsic = []
        for trap in self.traps:
            if isinstance(trap, FESTIM.ExtrinsicTrap):
                trap.create_form_density(dx, dt, T)
                self.extrinsic_formulations.append(trap.form_density)
        self.sub_expressions.extend(expressions_extrinsic)

    def solve_extrinsic_traps(self):
        for trap in self.traps:
            if isinstance(trap, FESTIM.ExtrinsicTrap):
                f.solve(trap.form_density == 0, trap.density[0], [])

    def update_extrinsic_traps_density(self):
        for trap in self.traps:
            if isinstance(trap, FESTIM.ExtrinsicTrap):
                trap.density_previous_solution.assign(trap.density[0])
