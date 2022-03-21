import FESTIM
import fenics as f


class TestCustomTrap:
    form_prms = {
        "prm1": 1,
        "prm2": 2
    }
    my_trap = FESTIM.CustomTrap(1, 1, 1, 1, 1, form_parameters=form_prms)
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
        """
        test to ensure the parameters within form_prms are contants
        """
        for prm in self.my_trap.form_parameters.values():
            assert isinstance(prm, (f.Expression, f.Constant))

    def test_create_form_density(self):
        """
        Checks that create_form_density produces the expected formulation
        """
        form_prms = self.my_trap.form_parameters
        density = self.my_trap.density[0]
        expected_form = (density - self.my_trap.density_previous_solution) /\
            self.dt.value * self.my_trap.density_test_function*self.dx
        expected_form += -form_prms["prm1"] * (self.my_temp.T +
                                               form_prms["prm2"]) *\
            self.my_trap.density_test_function*self.dx

        self.my_trap.create_form_density(self.dx, self.dt, self.my_temp)
        print(expected_form)
        print(self.my_trap.form_density)
        assert self.my_trap.form_density.equals(expected_form)
