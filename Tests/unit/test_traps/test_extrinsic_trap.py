import FESTIM
import fenics as f


class TestExtrinsicTrap:
    form_prms = {
        "phi_0": 1 + FESTIM.t + FESTIM.x,
        "n_amax": 2,
        "n_bmax": 2,
        "eta_a": 3,
        "eta_b": 4,
        "f_a": 5,
        "f_b": 6
    }
    my_trap = FESTIM.ExtrinsicTrap(1, 1, 1, 1, 1, form_parameters=form_prms)
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_trap.density = [f.Function(V)]
    my_trap.density_previous_solution = f.Function(V)
    my_trap.density_test_function = f.TestFunction(V)
    my_temp = FESTIM.Temperature(value=100)
    my_temp.T = f.Function(V, name="T")
    dx = f.dx()
    dt = FESTIM.Stepsize(initial_value=1)

    def test_that_form_parameters_are_expressions(self):
        for prm in self.my_trap.form_parameters.values():
            assert isinstance(prm, (f.Expression, f.Constant))

    def test_create_form_density(self):
        form_prms = self.my_trap.form_parameters
        density = self.my_trap.density[0]
        T = self.my_temp
        expected_form = (density - self.my_trap.density_previous_solution)/self.dt.value * self.my_trap.density_test_function*self.dx
        expected_form += -form_prms["phi_0"] * (
            (1 - density/form_prms["n_amax"])*form_prms["eta_a"]*form_prms["f_a"] +
            (1 - density/form_prms["n_bmax"])*form_prms["eta_b"]*form_prms["f_b"] ) * \
            self.my_trap.density_test_function*self.dx

        self.my_trap.create_form_density(self.dx, self.dt, T)
        print(expected_form)
        print(self.my_trap.form_density)
        assert self.my_trap.form_density.equals(expected_form)
