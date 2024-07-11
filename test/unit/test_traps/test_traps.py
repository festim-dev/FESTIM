import festim
import fenics as f
import pytest


def test_set_traps():
    my_mat = festim.Material(1, 1, 0)
    trap1 = festim.Trap(1, 1, 1, 1, [my_mat], density=1)
    trap2 = festim.Trap(2, 2, 2, 2, [my_mat], density=1)

    combinations = [[trap1], [trap1, trap2]]

    for trap_combination in combinations:
        festim.Traps(trap_combination)


def add_functions(trap, V, id=1):
    trap.solution = f.Function(V, name="c_t_{}".format(id))
    trap.previous_solution = f.Function(V, name="c_t_n_{}".format(id))
    trap.test_function = f.TestFunction(V)


class TestCreateTrappingForms:
    """
    General test for the create_forms method of the festim.Traps class
    """

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = festim.Mobile()
    my_mobile.solution = f.Function(V, name="c_m")
    my_mobile.previous_solution = f.Function(V, name="c_m_n")
    my_mobile.test_function = f.TestFunction(V)
    my_temp = festim.Temperature(value=100)
    my_temp.T = f.interpolate(f.Constant(100), V)
    dx = f.dx()
    dt = festim.Stepsize(initial_value=1)
    mat1 = festim.Material(1, D_0=1, E_D=1, S_0=2, E_S=3)
    mat2 = festim.Material(2, D_0=2, E_D=2, S_0=3, E_S=4)
    my_mats = festim.Materials([mat1, mat2])

    trap1 = festim.Trap(
        k_0=1, E_k=2, p_0=1, E_p=2, materials=[mat1, mat2], density=1 + festim.x
    )
    add_functions(trap1, V, id=1)
    trap2 = festim.Trap(
        k_0=2, E_k=3, p_0=1, E_p=2, materials=mat1, density=1 + festim.t
    )
    add_functions(trap2, V, id=2)

    def test_one_trap_steady_state(self):
        """Tests the case of one trap in steady-state"""
        my_traps = festim.Traps([self.trap1])

        my_traps.create_forms(self.my_mobile, self.my_mats, self.my_temp, self.dx)

        for trap in my_traps:
            assert trap.F is not None

    def test_one_trap_transient(self):
        """Tests the case of one trap in transient"""
        my_traps = festim.Traps([self.trap1])

        my_traps.create_forms(
            self.my_mobile, self.my_mats, self.my_temp, self.dx, dt=self.dt
        )

        for trap in my_traps:
            assert trap.F is not None

    def test_two_traps_transient(self):
        """Tests the case of two traps in transient"""
        my_traps = festim.Traps([self.trap1, self.trap2])

        my_traps.create_forms(
            self.my_mobile, self.my_mats, self.my_temp, self.dx, dt=self.dt
        )

        for trap in my_traps:
            assert trap.F is not None


class TestGetTrap:
    """
    General test for the get_trap method of the festim.Traps class
    """

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = festim.Mobile()
    my_mobile.solution = f.Function(V, name="c_m")
    my_mobile.previous_solution = f.Function(V, name="c_m_n")
    my_mobile.test_function = f.TestFunction(V)
    my_temp = festim.Temperature(value=100)
    my_temp.T = f.interpolate(f.Constant(100), V)
    dx = f.dx()
    dt = f.Constant(1)

    trap1 = festim.Trap(
        k_0=1, E_k=2, p_0=1, E_p=2, materials="mat_name", density=1, id=1
    )
    add_functions(trap1, V, id=1)
    trap2 = festim.Trap(
        k_0=2, E_k=3, p_0=1, E_p=2, materials="mat_name", density=1, id=2
    )
    add_functions(trap2, V, id=2)
    my_traps = festim.Traps([trap1, trap2])

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


class TestTrapsMethods:
    """Checks that festim.Traps methods work properly"""

    my_mat = festim.Material(1, 1, 0)
    my_trap1 = festim.Trap(1, 1, 1, 1, [my_mat], density=1)
    my_trap2 = festim.Trap(2, 1, 1, 1, [my_mat], density=1)

    my_traps = festim.Traps([my_trap1])

    def test_traps_append(self):
        self.my_traps.append(self.my_trap2)
        assert self.my_traps == [self.my_trap1, self.my_trap2]

    def test_traps_insert(self):
        self.my_traps.insert(0, self.my_trap2)
        assert self.my_traps == [self.my_trap2, self.my_trap1, self.my_trap2]

    def test_traps_setitem(self):
        self.my_traps[0] = self.my_trap1
        assert self.my_traps == [self.my_trap1, self.my_trap1, self.my_trap2]

    def test_traps_extend_list_type(self):
        self.my_traps.extend([self.my_trap1])
        assert self.my_traps == [
            self.my_trap1,
            self.my_trap1,
            self.my_trap2,
            self.my_trap1,
        ]

    def test_traps_extend_self_type(self):
        self.my_traps.extend(festim.Traps([self.my_trap2]))
        assert self.my_traps == festim.Traps(
            [self.my_trap1, self.my_trap1, self.my_trap2, self.my_trap1, self.my_trap2]
        )


def test_set_traps_wrong_type():
    """Checks an error is raised when festim.Traps is set with the wrong type"""
    my_mat = festim.Material(1, 1, 0)
    trap1 = festim.Trap(1, 1, 1, 1, [my_mat], density=1)

    combinations = [trap1, "coucou", 1, True]

    for trap_combination in combinations:
        with pytest.raises(
            TypeError,
            match="festim.Traps must be a list",
        ):
            festim.Traps(trap_combination)

    with pytest.raises(
        TypeError,
        match="festim.Traps must be a list of festim.Trap",
    ):
        festim.Traps([trap1, 2])


def test_assign_traps_wrong_type():
    """Checks an error is raised when the wrong type is assigned to festim.Traps"""
    my_mat = festim.Material(1, 1, 0)
    trap1 = festim.Trap(1, 1, 1, 1, [my_mat], density=1)
    my_traps = festim.Traps([trap1])

    combinations = ["coucou", 1, True]
    error_pattern = "festim.Traps must be a list of festim.Trap"

    for trap_combination in combinations:
        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_traps.append(trap_combination)

        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_traps.extend([trap_combination])

        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_traps[0] = trap_combination

        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_traps.insert(0, trap_combination)


class TestTrapsPropertyDeprWarn:
    """
    A temporary test to check DeprecationWarnings in festim.Traps.traps
    """

    my_mat = festim.Material(1, 1, 1)
    my_trap = festim.Trap(1, 1, 1, 1, [my_mat], 1)

    my_traps = festim.Traps([])

    def test_property_depr_warns(self):
        with pytest.deprecated_call():
            self.my_traps.traps

    def test_property_setter_depr_warns(self):
        with pytest.deprecated_call():
            self.my_traps.traps = [self.my_trap]


class TestTrapsPropertyRaiseError:
    """
    A temporary test to check TypeErrors in festim.Traps.traps
    """

    my_mat = festim.Material(1, 1, 1)
    my_trap = festim.Trap(1, 1, 1, 1, [my_mat], 1)

    my_traps = festim.Traps([])

    def test_set_traps_wrong_type(self):
        with pytest.raises(
            TypeError,
            match="traps must be a list",
        ):
            self.my_traps.traps = self.my_trap

    def test_set_traps_list_wrong_type(self):
        with pytest.raises(
            TypeError,
            match="traps must be a list of festim.Trap",
        ):
            self.my_traps.traps = [self.my_trap, 1]


def test_instanciate_with_no_elements():
    """
    Test to catch bug described in issue #724
    """
    # define exports
    festim.Traps()
