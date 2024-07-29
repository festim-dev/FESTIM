from festim import Trap, as_constant_or_expression
from fenics import NewtonSolver, MPI
import warnings


class ExtrinsicTrapBase(Trap):
    def __init__(
        self,
        k_0,
        E_k,
        p_0,
        E_p,
        materials,
        id=None,
        absolute_tolerance=1e0,
        relative_tolerance=1e-10,
        maximum_iterations=30,
        linear_solver=None,
        preconditioner=None,
        **kwargs,
    ):
        """Inits ExtrinsicTrap

        Args:
            E_k (float, list): trapping pre-exponential factor (m3 s-1)
            k_0 (float, list): trapping activation energy (eV)
            p_0 (float, list): detrapping pre-exponential factor (s-1)
            E_p (float, list): detrapping activation energy (eV)
            materials (list, int): the materials ids the trap is living in
            id (int, optional): The trap id. Defaults to None.
            absolute_tolerance (float, optional): the absolute tolerance of the newton
                solver. Defaults to 1e-0
            relative_tolerance (float, optional): the relative tolerance of the newton
                solver. Defaults to 1e-10
            maximum_iterations (int, optional): maximum iterations allowed for
                the solver to converge. Defaults to 30.
            linear_solver (str, optional): linear solver method for the newton solver,
                options can be viewed with print(list_linear_solver_methods()).
                If None, the default fenics linear solver will be used ("umfpack").
                More information can be found at: https://fenicsproject.org/pub/tutorial/html/._ftut1017.html.
                Defaults to None.
            preconditioner (str, optional): preconditioning method for the newton solver,
                options can be viewed by print(list_krylov_solver_preconditioners()).
                Defaults to "default".
        """
        super().__init__(k_0, E_k, p_0, E_p, materials, density=None, id=id)
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.maximum_iterations = maximum_iterations
        self.linear_solver = linear_solver
        self.preconditioner = preconditioner

        self.newton_solver = None
        for name, val in kwargs.items():
            setattr(self, name, as_constant_or_expression(val))
        self.density_previous_solution = None
        self.density_test_function = None

    @property
    def newton_solver(self):
        return self._newton_solver

    @newton_solver.setter
    def newton_solver(self, value):
        if value is None:
            self._newton_solver = value
        elif isinstance(value, NewtonSolver):
            if self._newton_solver:
                print("Settings for the Newton solver will be overwritten")
            self._newton_solver = value
        else:
            raise TypeError("accepted type for newton_solver is fenics.NewtonSolver")

    def define_newton_solver(self):
        """Creates the Newton solver and sets its parameters"""
        self.newton_solver = NewtonSolver(MPI.comm_world)
        self.newton_solver.parameters["error_on_nonconvergence"] = False
        self.newton_solver.parameters["absolute_tolerance"] = self.absolute_tolerance
        self.newton_solver.parameters["relative_tolerance"] = self.relative_tolerance
        self.newton_solver.parameters["maximum_iterations"] = self.maximum_iterations
        self.newton_solver.parameters["linear_solver"] = self.linear_solver
        self.newton_solver.parameters["preconditioner"] = self.preconditioner


class ExtrinsicTrap(ExtrinsicTrapBase):
    """
    For details in the forumation see
    http://www.sciencedirect.com/science/article/pii/S2352179119300547

    Args:
        E_k (float, list): trapping pre-exponential factor (m3 s-1)
        k_0 (float, list): trapping activation energy (eV)
        p_0 (float, list): detrapping pre-exponential factor (s-1)
        E_p (float, list): detrapping activation energy (eV)
        materials (list, int): the materials ids the trap is living in
        id (int, optional): The trap id. Defaults to None.
    """

    def __init__(
        self,
        k_0,
        E_k,
        p_0,
        E_p,
        materials,
        phi_0,
        n_amax,
        n_bmax,
        eta_a,
        eta_b,
        f_a,
        f_b,
        id=None,
        **kwargs,
    ):
        super().__init__(
            k_0,
            E_k,
            p_0,
            E_p,
            materials,
            phi_0=phi_0,
            n_amax=n_amax,
            n_bmax=n_bmax,
            eta_a=eta_a,
            eta_b=eta_b,
            f_a=f_a,
            f_b=f_b,
            id=id,
            **kwargs,
        )

    def create_form_density(self, dx, dt, T):
        """
        Creates the variational formulation for the extrinsic trap density.

        Args:
            dx (fenics.Measure): the dx measure of the sim
            dt (festim.Stepsize): the stepsize of the simulation.
            T (festim.Temperature): the temperature of the
                simulation

        .. note::
            T is an argument, although is not used in the formulation of
            extrinsic traps, but potential for subclasses of extrinsic traps
        """
        density = self.density[0]
        F = (
            ((density - self.density_previous_solution) / dt.value)
            * self.density_test_function
            * dx
        )
        F += (
            -self.phi_0
            * (
                (1 - density / self.n_amax) * self.eta_a * self.f_a
                + (1 - density / self.n_bmax) * self.eta_b * self.f_b
            )
            * self.density_test_function
            * dx
        )
        self.form_density = F
