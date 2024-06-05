import festim
import fenics as f


class TestExtrinsicTrap:
    """
    General test for the ExtrinsicTrap class
    """

    my_trap = festim.ExtrinsicTrap(
        1,
        1,
        1,
        1,
        "mat_name",
        phi_0=1 + festim.t + festim.x,
        n_amax=2,
        n_bmax=2,
        eta_a=3,
        eta_b=4,
        f_a=5,
        f_b=6,
    )
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_trap.density = [f.Function(V)]
    my_trap.density_previous_solution = f.Function(V)
    my_trap.density_test_function = f.TestFunction(V)
    my_temp = festim.Temperature(value=100)
    my_temp.T = f.Function(V, name="T")
    dx = f.dx()
    dt = festim.Stepsize(initial_value=1)

    def test_that_form_parameters_are_expressions(self):
        """
        Checks that the attributes are of correct types
        """
        prms = [
            self.my_trap.phi_0,
            self.my_trap.n_amax,
            self.my_trap.n_bmax,
            self.my_trap.n_bmax,
            self.my_trap.eta_a,
            self.my_trap.f_a,
            self.my_trap.f_b,
        ]
        for prm in prms:
            assert isinstance(prm, (f.Expression, f.Constant))

    def test_create_form_density(self):
        """
        Checks that the forumlation produced by the create_form_density
        function produces the expected formulation
        """
        density = self.my_trap.density[0]
        T = self.my_temp
        expected_form = (
            (density - self.my_trap.density_previous_solution)
            / self.dt.value
            * self.my_trap.density_test_function
            * self.dx
        )
        expected_form += (
            -self.my_trap.phi_0
            * (
                (1 - density / self.my_trap.n_amax)
                * self.my_trap.eta_a
                * self.my_trap.f_a
                + (1 - density / self.my_trap.n_bmax)
                * self.my_trap.eta_b
                * self.my_trap.f_b
            )
            * self.my_trap.density_test_function
            * self.dx
        )

        self.my_trap.create_form_density(self.dx, self.dt, T)
        print(expected_form)
        print(self.my_trap.form_density)
        assert self.my_trap.form_density.equals(expected_form)

    def test_solver_parameters(self):
        """
        A test to ensure the extrinsic trap solver parameters can be accessed
        """
        self.my_trap.absolute_tolerance = 1
        self.my_trap.relative_tolerance = 1
        self.my_trap.maximum_iterations = 1
        self.my_trap.linear_solver = "mumps"
        self.my_trap.preconditioner = "icc"
        self.my_trap.newton_solver = f.NewtonSolver()
