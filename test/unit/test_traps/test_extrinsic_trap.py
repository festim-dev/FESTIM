import festim
import fenics as f


class TestExtrinsicTrap:

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
