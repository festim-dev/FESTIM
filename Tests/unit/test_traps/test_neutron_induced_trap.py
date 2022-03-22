import FESTIM
import fenics as f


class TestNeutronInducedTrap:
    """
    General test for the NeutronInducedTrap class
    """
    form_prms = {
        "phi": 1,
        "K": 1,
        "n_max": 2,
        "A_0": 1,
        "E_A": 1,
    }
    my_trap = FESTIM.NeutronInducedTrap(1, 1, 1, 1, 1,
                                        form_parameters=form_prms)
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_trap.density = [f.Function(V)]
    my_trap.density_previous_solution = f.Function(V)
    my_trap.density_test_function = f.TestFunction(V)
    my_temp = FESTIM.Temperature(value=100)
    my_temp.T = f.Function(V, name="T")
    dx = f.dx()
    dt = FESTIM.Stepsize(initial_value=1)

    def test_create_form_density(self):
        """
        Checks that the forumlation produced by the create_form_density
        function produces the expected formulation
        """
        phi = self.my_trap.form_parameters["phi"]
        K = self.my_trap.form_parameters["K"]
        n_max = self.my_trap.form_parameters["n_max"]
        A_0 = self.my_trap.form_parameters["A_0"]
        E_A = self.my_trap.form_parameters["E_A"]
        T = self.my_temp.T
        density = self.my_trap.density[0]

        expected_form = (density - self.my_trap.density_previous_solution) /\
            self.dt.value * self.my_trap.density_test_function*self.dx
        expected_form += -phi*K*(1 - (density/n_max)) *\
            self.my_trap.density_test_function*self.dx
        expected_form += A_0*f.exp(-E_A/(FESTIM.k_B*T))*density *\
            self.my_trap.density_test_function*self.dx
        self.my_trap.create_form_density(self.dx, self.dt, self.my_temp)
        print(expected_form)
        print(self.my_trap.form_density)
        assert self.my_trap.form_density.equals(expected_form)


class TestNeutronInducedTrapVaryingTime:
    """
    Test for NeutronInducedTrap class, with varying phi with time
    """
    form_prms = {
        "phi": 100+20*FESTIM.t,
        "K": 1,
        "n_max": 2,
        "A_0": 1,
        "E_A": 1,
    }
    my_trap = FESTIM.NeutronInducedTrap(1, 1, 1, 1, 1,
                                        form_parameters=form_prms)
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_trap.density = [f.Function(V)]
    my_trap.density_previous_solution = f.Function(V)
    my_trap.density_test_function = f.TestFunction(V)
    my_temp = FESTIM.Temperature(value=100)
    my_temp.T = f.Function(V, name="T")
    dx = f.dx()
    dt = FESTIM.Stepsize(initial_value=1)

    def test_create_form_density(self):
        """
        Creates the formulation for the density function using parameters
        defined in my_trap and my_temps
        """
        phi = self.my_trap.form_parameters["phi"]
        K = self.my_trap.form_parameters["K"]
        n_max = self.my_trap.form_parameters["n_max"]
        A_0 = self.my_trap.form_parameters["A_0"]
        E_A = self.my_trap.form_parameters["E_A"]
        T = self.my_temp.T
        density = self.my_trap.density[0]

        expected_form = (density - self.my_trap.density_previous_solution) /\
            self.dt.value * self.my_trap.density_test_function*self.dx
        expected_form += -phi*K*(1 - (density/n_max)) *\
            self.my_trap.density_test_function*self.dx
        expected_form += A_0*f.exp(-E_A/(FESTIM.k_B*T))*density *\
            self.my_trap.density_test_function*self.dx
        self.my_trap.create_form_density(self.dx, self.dt, self.my_temp)
        print(expected_form)
        print(self.my_trap.form_density)
        assert self.my_trap.form_density.equals(expected_form)


class TestNeutronInducedTrapVaryingPhiSpatially:
    """
    Test for NeutronInducedTrap class, with varying phi with x
    """
    form_prms = {
        "phi": 100+20*FESTIM.x,
        "K": 1,
        "n_max": 2,
        "A_0": 1,
        "E_A": 1,
    }
    my_trap = FESTIM.NeutronInducedTrap(1, 1, 1, 1, 1,
                                        form_parameters=form_prms)
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_trap.density = [f.Function(V)]
    my_trap.density_previous_solution = f.Function(V)
    my_trap.density_test_function = f.TestFunction(V)
    my_temp = FESTIM.Temperature(value=100)
    my_temp.T = f.Function(V, name="T")
    dx = f.dx()
    dt = FESTIM.Stepsize(initial_value=1)

    def test_create_form_density(self):
        """
        Creates the formulation for the density function using paramters
        defined in my_trap and my_temps
        """
        phi = self.my_trap.form_parameters["phi"]
        K = self.my_trap.form_parameters["K"]
        n_max = self.my_trap.form_parameters["n_max"]
        A_0 = self.my_trap.form_parameters["A_0"]
        E_A = self.my_trap.form_parameters["E_A"]
        T = self.my_temp.T
        density = self.my_trap.density[0]

        expected_form = (density - self.my_trap.density_previous_solution) /\
            self.dt.value * self.my_trap.density_test_function*self.dx
        expected_form += -phi*K*(1 - (density/n_max)) *\
            self.my_trap.density_test_function*self.dx
        expected_form += A_0*f.exp(-E_A/(FESTIM.k_B*T))*density *\
            self.my_trap.density_test_function*self.dx
        self.my_trap.create_form_density(self.dx, self.dt, self.my_temp)
        print(expected_form)
        print(self.my_trap.form_density)
        assert self.my_trap.form_density.equals(expected_form)


class TestNeutronInducedTrapTempVaryingTime:
    """
    Test for NeutronInducedTrap class, with varying temperature with time
    """
    form_prms = {
        "phi": 100,
        "K": 1,
        "n_max": 2,
        "A_0": 1,
        "E_A": 1,
    }
    my_trap = FESTIM.NeutronInducedTrap(1, 1, 1, 1, 1,
                                        form_parameters=form_prms)
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_trap.density = [f.Function(V)]
    my_trap.density_previous_solution = f.Function(V)
    my_trap.density_test_function = f.TestFunction(V)
    my_temp = FESTIM.Temperature(value=100 + 50*FESTIM.t)
    my_temp.T = f.Function(V, name="T")
    dx = f.dx()
    dt = FESTIM.Stepsize(initial_value=1)

    def test_create_form_density(self):
        """
        Creates the formulation for the density function using paramters
        defined in my_trap and my_temps
        """
        phi = self.my_trap.form_parameters["phi"]
        K = self.my_trap.form_parameters["K"]
        n_max = self.my_trap.form_parameters["n_max"]
        A_0 = self.my_trap.form_parameters["A_0"]
        E_A = self.my_trap.form_parameters["E_A"]
        T = self.my_temp.T
        density = self.my_trap.density[0]

        expected_form = (density - self.my_trap.density_previous_solution) /\
            self.dt.value * self.my_trap.density_test_function*self.dx
        expected_form += -phi*K*(1 - (density/n_max)) *\
            self.my_trap.density_test_function*self.dx
        expected_form += A_0*f.exp(-E_A/(FESTIM.k_B*T))*density *\
            self.my_trap.density_test_function*self.dx
        self.my_trap.create_form_density(self.dx, self.dt, self.my_temp)
        print(expected_form)
        print(self.my_trap.form_density)
        assert self.my_trap.form_density.equals(expected_form)
