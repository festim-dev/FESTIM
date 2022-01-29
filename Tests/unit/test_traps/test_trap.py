import FESTIM
import fenics as f
import pytest


def add_functions(trap, V, id=1):
    trap.solution = f.Function(V, name="c_t_{}".format(id))
    trap.previous_solution = f.Function(V, name="c_t_n_{}".format(id))
    trap.test_function = f.TestFunction(V)


def test_error_if_duplicate_material():
    with pytest.raises(ValueError, match="Duplicate materials in trap"):
        FESTIM.Trap(1, 1, 1, 1, [1, 1], 1)


class TestCreateTrappingForm:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = FESTIM.Mobile()
    my_mobile.solution = f.Function(V, name="c_m")
    my_mobile.previous_solution = f.Function(V, name="c_m_n")
    my_mobile.test_function = f.TestFunction(V)
    my_temp = FESTIM.Temperature(value=100)
    my_temp.create_functions(V)
    dx = f.dx()
    dt = FESTIM.Stepsize(initial_value=1)

    mat1 = FESTIM.Material(1, D_0=1, E_D=1, S_0=2, E_S=3)
    mat2 = FESTIM.Material(2, D_0=2, E_D=2, S_0=3, E_S=4)

    def test_steady_state(self):
        # build
        my_trap = FESTIM.Trap(
            k_0=1, E_k=2, p_0=3, E_p=4,
            materials=1, density=1 + FESTIM.x)
        my_trap.F = 0
        my_trap.solution = f.Function(self.V, name="c_t")
        my_trap.previous_solution = f.Function(self.V, name="c_t_n")
        my_trap.test_function = f.TestFunction(self.V)
        my_mats = FESTIM.Materials([self.mat1])

        # run
        my_trap.create_trapping_form(self.my_mobile, my_mats, self.my_temp, self.dx)

        # test
        v = my_trap.test_function
        expected_form = - my_trap.k_0 * f.exp(-my_trap.E_k/FESTIM.k_B/self.my_temp.T) * self.my_mobile.solution \
            * (my_trap.density[0] - my_trap.solution) * \
            v*self.dx(1)
        expected_form += my_trap.p_0*f.exp(-my_trap.E_p/FESTIM.k_B/self.my_temp.T)*my_trap.solution * \
            v*self.dx(1)
        print("expected F:", expected_form)
        print("produced F_trapping:", my_trap.F_trapping)
        print("produced F:", my_trap.F)
        assert my_trap.F.equals(expected_form)
        assert my_trap.F_trapping.equals(expected_form)

    def test_transient(self):
        # build
        my_trap = FESTIM.Trap(
            k_0=1, E_k=2, p_0=3, E_p=4,
            materials=1, density=1 + FESTIM.x)
        my_trap.F = 0
        my_trap.solution = f.Function(self.V, name="c_t")
        my_trap.previous_solution = f.Function(self.V, name="c_t_n")
        my_trap.test_function = f.TestFunction(self.V)

        my_mats = FESTIM.Materials([self.mat1])

        # run
        my_trap.create_trapping_form(
            self.my_mobile, my_mats, self.my_temp, self.dx, dt=self.dt)

        # test
        v = my_trap.test_function
        expected_form = ((my_trap.solution - my_trap.previous_solution) / self.dt.value) * my_trap.test_function * self.dx
        expected_form += - my_trap.k_0 * f.exp(-my_trap.E_k/FESTIM.k_B/self.my_temp.T) * self.my_mobile.solution \
            * (my_trap.density[0] - my_trap.solution) * \
            v*self.dx(1)
        expected_form += my_trap.p_0*f.exp(-my_trap.E_p/FESTIM.k_B/self.my_temp.T)*my_trap.solution * \
            v*self.dx(1)
        print("expected F:", expected_form)
        print("produced F_trapping:", my_trap.F_trapping)
        print("produced F:", my_trap.F)
        assert my_trap.F.equals(expected_form)
        assert my_trap.F_trapping.equals(expected_form)

    def test_chemical_potential(self):
        # build
        my_trap = FESTIM.Trap(
            k_0=1, E_k=2, p_0=3, E_p=4,
            materials=1, density=1 + FESTIM.x)
        my_trap.F = 0
        my_trap.solution = f.Function(self.V, name="c_t")
        my_trap.previous_solution = f.Function(self.V, name="c_t_n")
        my_trap.test_function = f.TestFunction(self.V)
        my_mats = FESTIM.Materials([self.mat1])

        # run
        my_trap.create_trapping_form(self.my_mobile, my_mats, self.my_temp, self.dx, chemical_pot=True)

        # test
        v = my_trap.test_function
        c_0 = self.my_mobile.solution*self.mat1.S_0*f.exp(-self.mat1.E_S/FESTIM.k_B/self.my_temp.T)
        expected_form = - my_trap.k_0 * f.exp(-my_trap.E_k/FESTIM.k_B/self.my_temp.T) * c_0 \
            * (my_trap.density[0] - my_trap.solution) * \
            v*self.dx(1)
        expected_form += my_trap.p_0*f.exp(-my_trap.E_p/FESTIM.k_B/self.my_temp.T)*my_trap.solution * \
            v*self.dx(1)
        print("expected F:")
        print(expected_form)
        print("produced F_trapping:")
        print(my_trap.F_trapping)
        print("produced F:")
        print(my_trap.F)
        assert my_trap.F.equals(expected_form)
        assert my_trap.F_trapping.equals(expected_form)

    def test_2_materials(self):
        # build
        my_trap = FESTIM.Trap(
            k_0=1, E_k=2, p_0=3, E_p=4,
            materials=[1, 2], density=1 + FESTIM.x)
        my_trap.F = 0
        my_trap.solution = f.Function(self.V, name="c_t")
        my_trap.previous_solution = f.Function(self.V, name="c_t_n")
        my_trap.test_function = f.TestFunction(self.V)
        my_mats = FESTIM.Materials([self.mat1, self.mat2])

        # run
        my_trap.create_trapping_form(self.my_mobile, my_mats, self.my_temp, self.dx)

        # test
        v = my_trap.test_function
        expected_form = 0
        for mat_id in my_trap.materials:
            expected_form += - my_trap.k_0 * f.exp(-my_trap.E_k/FESTIM.k_B/self.my_temp.T) * self.my_mobile.solution \
                * (my_trap.density[0] - my_trap.solution) * \
                v*self.dx(mat_id)
            expected_form += my_trap.p_0*f.exp(-my_trap.E_p/FESTIM.k_B/self.my_temp.T)*my_trap.solution * \
                v*self.dx(mat_id)

        print("expected F:", expected_form)
        print("produced F_trapping:", my_trap.F_trapping)
        print("produced F:", my_trap.F)
        assert my_trap.F.equals(expected_form)
        assert my_trap.F_trapping.equals(expected_form)

    def test_multi_parameters_trap(self):
        # build
        my_trap = FESTIM.Trap(
            k_0=[1, 2], E_k=[2, 2], p_0=[3, 2], E_p=[4, 2],
            materials=[1, 2], density=[1 + FESTIM.x, 1 + FESTIM.t])
        my_trap.sources = [
            FESTIM.Source(2 + FESTIM.x + FESTIM.t, volume=1, field="1")
        ]
        my_trap.F = 0
        add_functions(my_trap, self.V, id=1)
        my_mats = FESTIM.Materials([self.mat1, self.mat2])

        # run
        my_trap.create_trapping_form(self.my_mobile, my_mats, self.my_temp, self.dx)

        # test
        v = my_trap.test_function
        expected_form = 0
        for i in range(2):
            expected_form += - my_trap.k_0[i] * f.exp(-my_trap.E_k[i]/FESTIM.k_B/self.my_temp.T) * self.my_mobile.solution \
                * (my_trap.density[i] - my_trap.solution) * \
                v*self.dx(my_trap.materials[i])
            expected_form += my_trap.p_0[i]*f.exp(-my_trap.E_p[i]/FESTIM.k_B/self.my_temp.T)*my_trap.solution * \
                v*self.dx(my_trap.materials[i])

        print("expected F:", expected_form)
        print("produced F_trapping:", my_trap.F_trapping)
        print("produced F:", my_trap.F)
        assert my_trap.F.equals(expected_form)
        assert my_trap.F_trapping.equals(expected_form)

    def test_steady_state_trap_not_defined_everywhere(self):
        # build
        my_trap = FESTIM.Trap(
            k_0=1, E_k=2, p_0=3, E_p=4,
            materials=self.mat1.id, density=1 + FESTIM.x)
        my_trap.sources = [
            FESTIM.Source(2 + FESTIM.x + FESTIM.t, volume=1, field="1")
        ]
        my_trap.F = 0
        add_functions(my_trap, self.V, id=1)
        my_mats = FESTIM.Materials([self.mat1, self.mat2])

        # run
        my_trap.create_trapping_form(self.my_mobile, my_mats, self.my_temp, self.dx)

        # test
        v = my_trap.test_function
        expected_form = 0
        expected_form += - my_trap.k_0 * f.exp(-my_trap.E_k/FESTIM.k_B/self.my_temp.T) * self.my_mobile.solution * \
            (my_trap.density[0] - my_trap.solution) * v * self.dx(self.mat1.id)
        expected_form += my_trap.p_0*f.exp(-my_trap.E_p/FESTIM.k_B/self.my_temp.T)*my_trap.solution * \
            v*self.dx(self.mat1.id)
        expected_form += my_trap.solution * v * self.dx(self.mat2.id)
        print("expected F:", expected_form)
        print("produced F_trapping:", my_trap.F_trapping)
        print("produced F:", my_trap.F)
        assert my_trap.F.equals(expected_form)
        assert my_trap.F_trapping.equals(expected_form)


class TestCreateSourceForm:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = FESTIM.Mobile()
    my_mobile.solution = f.Function(V, name="c_m")
    my_mobile.previous_solution = f.Function(V, name="c_m_n")
    my_mobile.test_function = f.TestFunction(V)
    my_temp = FESTIM.Temperature(value=100)
    my_temp.create_functions(V)
    dx = f.dx()
    dt = FESTIM.Stepsize(initial_value=1)

    mat1 = FESTIM.Material(1, D_0=1, E_D=1, S_0=2, E_S=3)
    mat2 = FESTIM.Material(2, D_0=2, E_D=2, S_0=3, E_S=4)

    def test(self):
        # build
        my_trap = FESTIM.Trap(k_0=1, E_k=2, p_0=3, E_p=4, materials=1, density=1)
        my_trap.sources = [
            FESTIM.Source(2 + FESTIM.x + FESTIM.t, volume=1, field="1")
        ]
        my_trap.F = 0
        my_trap.test_function = f.TestFunction(self.V)

        # run
        my_trap.create_source_form(self.dx)

        # test
        source = my_trap.sub_expressions[0]
        v = my_trap.test_function
        expected_form = - source*v*self.dx(1)
        assert my_trap.F.equals(expected_form)
        assert my_trap.F_source.equals(expected_form)


class TestCreateForm:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = FESTIM.Mobile()
    my_mobile.solution = f.Function(V, name="c_m")
    my_mobile.previous_solution = f.Function(V, name="c_m_n")
    my_mobile.test_function = f.TestFunction(V)
    my_temp = FESTIM.Temperature(value=100)
    my_temp.create_functions(V)
    dx = f.dx()
    dt = FESTIM.Stepsize(initial_value=1)

    mat1 = FESTIM.Material(1, D_0=1, E_D=1, S_0=2, E_S=3)
    mat2 = FESTIM.Material(2, D_0=2, E_D=2, S_0=3, E_S=4)
    my_mats = FESTIM.Materials([mat1, mat2])

    def test_form_is_zero_by_default(self):
        my_empty_mats = FESTIM.Materials([])
        my_trap = FESTIM.Trap(1, 1, 1, 1, [], 1)

        my_trap.create_form(self.my_mobile, my_empty_mats, self.my_temp, self.dx)

        assert my_trap.F == 0

    def test_1_mat_steady(self):
        # build
        my_trap = FESTIM.Trap(1, 1, 1, 1, materials=1, density=1)
        add_functions(my_trap, self.V, id=1)
        my_trap.F = 0

        my_trap.create_trapping_form(self.my_mobile, self.my_mats, self.my_temp, self.dx)
        expected_form = my_trap.F
        # run
        my_trap.create_form(self.my_mobile, self.my_mats, self.my_temp, self.dx)

        # test
        print(my_trap.F)
        print(expected_form)
        assert my_trap.F.equals(expected_form)

    def test_1_mat_transient(self):
        # build
        my_trap = FESTIM.Trap(1, 1, 1, 1, materials=1, density=1)
        add_functions(my_trap, self.V, id=1)
        my_trap.F = 0

        my_trap.create_trapping_form(self.my_mobile, self.my_mats, self.my_temp, self.dx, self.dt)
        expected_form = my_trap.F
        # run
        my_trap.create_form(self.my_mobile, self.my_mats, self.my_temp, self.dx, self.dt)

        # test
        print(my_trap.F)
        print(expected_form)
        assert my_trap.F.equals(expected_form)

    def test_2_mats_transient(self):
        # build
        my_trap = FESTIM.Trap(1, 1, 1, 1, materials=[1, 2], density=1)
        add_functions(my_trap, self.V, id=1)
        my_trap.F = 0

        my_trap.create_trapping_form(self.my_mobile, self.my_mats, self.my_temp, self.dx, self.dt)
        expected_form = my_trap.F
        # run
        my_trap.create_form(self.my_mobile, self.my_mats, self.my_temp, self.dx, self.dt)

        # test
        print(my_trap.F)
        print(expected_form)
        assert my_trap.F.equals(expected_form)

    def test_1_mat_and_source(self):
        # build
        my_trap = FESTIM.Trap(1, 1, 1, 1, materials=2, density=1)
        my_trap.sources = [
            FESTIM.Source(1 + FESTIM.x + FESTIM.y, volume=1, field="1")
        ]
        add_functions(my_trap, self.V, id=1)
        my_trap.F = 0
        my_trap.create_trapping_form(self.my_mobile, self.my_mats, self.my_temp, self.dx, self.dt)
        expected_form = my_trap.F
        # run
        my_trap.create_form(self.my_mobile, self.my_mats, self.my_temp, self.dx, self.dt)

        # test
        expected_form += my_trap.F_source
        print(my_trap.F)
        print(expected_form)
        assert my_trap.F.equals(expected_form)
