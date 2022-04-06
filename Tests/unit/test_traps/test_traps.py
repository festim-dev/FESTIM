import FESTIM
import fenics as f
import pytest


def add_functions(trap, V, id=1):
    trap.solution = f.Function(V, name="c_t_{}".format(id))
    trap.previous_solution = f.Function(V, name="c_t_n_{}".format(id))
    trap.test_function = f.TestFunction(V)


class TestCreateTrappingForms:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = FESTIM.Mobile()
    my_mobile.solution = f.Function(V, name="c_m")
    my_mobile.previous_solution = f.Function(V, name="c_m_n")
    my_mobile.test_function = f.TestFunction(V)
    my_temp = FESTIM.Temperature(value=100)
    my_temp.T = f.interpolate(f.Constant(100), V)
    dx = f.dx()
    dt = FESTIM.Stepsize(initial_value=1)
    mat1 = FESTIM.Material(1, D_0=1, E_D=1, S_0=2, E_S=3)
    mat2 = FESTIM.Material(2, D_0=2, E_D=2, S_0=3, E_S=4)
    my_mats = FESTIM.Materials([mat1, mat2])

    trap1 = FESTIM.Trap(
        k_0=1, E_k=2, p_0=1, E_p=2, materials=[1, 2], density=1 + FESTIM.x
    )
    add_functions(trap1, V, id=1)
    trap2 = FESTIM.Trap(k_0=2, E_k=3, p_0=1, E_p=2, materials=1, density=1 + FESTIM.t)
    add_functions(trap2, V, id=2)

    def test_one_trap_steady_state(self):
        my_traps = FESTIM.Traps([self.trap1])

        my_traps.create_forms(self.my_mobile, self.my_mats, self.my_temp, self.dx)

        for trap in my_traps.traps:
            assert trap.F is not None

    def test_one_trap_transient(self):
        my_traps = FESTIM.Traps([self.trap1])

        my_traps.create_forms(
            self.my_mobile, self.my_mats, self.my_temp, self.dx, dt=self.dt
        )

        for trap in my_traps.traps:
            assert trap.F is not None

    def test_two_traps_transient(self):
        my_traps = FESTIM.Traps([self.trap1, self.trap2])

        my_traps.create_forms(
            self.my_mobile, self.my_mats, self.my_temp, self.dx, dt=self.dt
        )

        for trap in my_traps.traps:
            assert trap.F is not None


class TestGetTrap:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = FESTIM.Mobile()
    my_mobile.solution = f.Function(V, name="c_m")
    my_mobile.previous_solution = f.Function(V, name="c_m_n")
    my_mobile.test_function = f.TestFunction(V)
    my_temp = FESTIM.Temperature(value=100)
    my_temp.T = f.interpolate(f.Constant(100), V)
    dx = f.dx()
    dt = f.Constant(1)

    trap1 = FESTIM.Trap(k_0=1, E_k=2, p_0=1, E_p=2, materials=1, density=1)
    add_functions(trap1, V, id=1)
    trap2 = FESTIM.Trap(k_0=2, E_k=3, p_0=1, E_p=2, materials=1, density=1)
    add_functions(trap2, V, id=2)
    my_traps = FESTIM.Traps([trap1, trap2])

    def test_trap_is_found(self):
        assert self.my_traps.get_trap(id=1) == self.trap1
        assert self.my_traps.get_trap(id=2) == self.trap2

    def test_error_is_raised_when_not_found(self):
        id = 3
        with pytest.raises(ValueError, match="Couldn't find trap {}".format(id)):
            self.my_traps.get_trap(id=id)

        id = -2
        with pytest.raises(ValueError, match="Couldn't find trap {}".format(id)):
            self.my_traps.get_trap(id=id)
