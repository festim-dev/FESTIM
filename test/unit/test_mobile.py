import festim
import fenics as f
from ufl.core.multiindex import Index


def test_mobile_create_diffusion_form():
    """Tests the create_diffusion_form method of the festim.Mobile class"""
    # build
    Index._globalcount = 8
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = festim.Mobile()
    my_mobile.F = 0
    my_mobile.solution = f.Function(V)
    my_mobile.previous_solution = f.Function(V)
    my_mobile.test_function = f.TestFunction(V)

    mat = festim.Material(1, D_0=1, E_D=1)
    my_mats = festim.Materials([mat])
    dx = f.dx()
    my_mesh = festim.Mesh(mesh)
    my_mesh.dx = dx
    T = festim.Temperature(value=100)
    T.T = f.interpolate(f.Constant(100), V)
    # run
    my_mobile.create_diffusion_form(my_mats, my_mesh, T)

    # test
    c_0 = my_mobile.solution
    v = my_mobile.test_function
    Index._globalcount = 8
    expected_form = f.dot(
        mat.D_0 * f.exp(-mat.E_D / festim.k_B / T.T) * f.grad(c_0), f.grad(v)
    ) * dx(1)
    assert my_mobile.F.equals(expected_form)
    assert my_mobile.F_diffusion.equals(expected_form)


def test_mobile_create_source_form_one_dict():
    """
    Tests the create_source_form method of the festim.Mobile class
    for the case of one festim.Source object
    """
    # build
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = festim.Mobile()
    my_mobile.F = 0
    my_mobile.test_function = f.TestFunction(V)

    dx = f.dx()
    my_mobile.sources = [festim.Source(2 + festim.x + festim.t, volume=1, field="0")]
    # run
    my_mobile.create_source_form(dx)

    # test
    source = my_mobile.sub_expressions[0]
    v = my_mobile.test_function
    expected_form = -source * v * dx(1)
    assert my_mobile.F.equals(expected_form)
    assert my_mobile.F_source.equals(expected_form)


def test_mobile_create_source_form_several_sources():
    """
    Tests the create_source_form method of the festim.Mobile class
    for the case of several festim.Source objects
    """
    # build
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = festim.Mobile()
    my_mobile.F = 0
    my_mobile.test_function = f.TestFunction(V)

    dx = f.dx()
    my_mobile.sources = [
        festim.Source(2 + festim.x + festim.t, volume=1, field="0"),
        festim.Source(1 + festim.x + festim.t, volume=2, field="0"),
        festim.Source(1 + festim.x + festim.t, volume=3, field="0"),
    ]
    # run
    my_mobile.create_source_form(dx)

    # test
    v = my_mobile.test_function
    expected_form = -my_mobile.sub_expressions[0] * v * dx(1)
    expected_form += -my_mobile.sub_expressions[1] * v * dx(2)
    expected_form += -my_mobile.sub_expressions[2] * v * dx(3)
    assert my_mobile.F.equals(expected_form)
    assert my_mobile.F_source.equals(expected_form)


def test_mobile_create_form():
    """Tests the create_form method of the festim.Mobile class"""
    # build
    Index._globalcount = 8
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = festim.Mobile()
    my_mobile.F = 0
    my_mobile.solution = f.Function(V)
    my_mobile.previous_solution = f.Function(V)
    my_mobile.test_function = f.TestFunction(V)
    my_mobile.source_term = {"value": 2 + festim.x + festim.t}

    mat = festim.Material(1, D_0=1, E_D=1)
    my_mats = festim.Materials([mat])
    mesh = festim.Mesh()
    mesh.dx = f.dx()
    mesh.ds = f.ds()
    T = festim.Temperature(value=100)
    T.T = f.interpolate(f.Constant(100), V)

    # run
    my_mobile.create_form(my_mats, mesh, T)

    # test
    Index._globalcount = 8
    expected_form = my_mobile.F_diffusion + my_mobile.F_source
    assert my_mobile.F.equals(expected_form)


def add_functions(trap, V, id=1):
    trap.solution = f.Function(V, name="c_t_{}".format(id))
    trap.previous_solution = f.Function(V, name="c_t_n_{}".format(id))
    trap.test_function = f.TestFunction(V)


class TestCreateDiffusionForm:
    """General test for the create_diffusion_form method of the festim.Mobile class"""

    mesh = f.UnitIntervalMesh(10)
    my_mesh = festim.Mesh(mesh)
    my_temp = festim.Temperature(value=100)
    my_temp.create_functions(my_mesh)
    my_mesh.dx = f.dx()
    dt = festim.Stepsize(initial_value=1)
    V = f.FunctionSpace(my_mesh.mesh, "CG", 1)
    mat1 = festim.Material(1, D_0=1, E_D=1, S_0=2, E_S=3)
    mat2 = festim.Material(2, D_0=2, E_D=2, S_0=3, E_S=4)

    def test_with_traps_transient(self):
        """Check for the case of transient simulation with traps"""
        # build
        Index._globalcount = 8
        my_mobile = festim.Mobile()
        my_mobile.F = 0
        my_mobile.solution = f.Function(self.V, name="c_m")
        my_mobile.previous_solution = f.Function(self.V, name="c_m_n")
        my_mobile.test_function = f.TestFunction(self.V)
        my_mats = festim.Materials([self.mat1])

        trap1 = festim.Trap(1, 1, 1, 1, self.mat1, 1)
        add_functions(trap1, self.V, id=1)
        trap2 = festim.Trap(2, 2, 2, 2, self.mat1, 2)
        add_functions(trap2, self.V, id=1)

        my_traps = festim.Traps([trap1, trap2])

        # run
        my_mobile.create_diffusion_form(
            my_mats, self.my_mesh, self.my_temp, dt=self.dt, traps=my_traps
        )

        # test
        Index._globalcount = 8
        v = my_mobile.test_function
        D = self.mat1.D_0 * f.exp(-self.mat1.E_D / festim.k_B / self.my_temp.T)
        c_0 = my_mobile.solution
        c_0_n = my_mobile.previous_solution
        expected_form = ((c_0 - c_0_n) / self.dt.value) * v * self.my_mesh.dx(1)
        expected_form += f.dot(D * f.grad(c_0), f.grad(v)) * self.my_mesh.dx(1)
        form_trapping_expected = 0
        for trap in my_traps:
            form_trapping_expected += (
                (
                    -trap.k_0
                    * f.exp(-trap.E_k / festim.k_B / self.my_temp.T)
                    * c_0
                    * (trap.density[0] - trap.solution)
                )
                * v
                * self.my_mesh.dx(1)
            )
            form_trapping_expected += (
                trap.p_0
                * f.exp(-trap.E_p / festim.k_B / self.my_temp.T)
                * trap.solution
                * v
                * self.my_mesh.dx(1)
            )
        expected_form += -form_trapping_expected
        print("expected F:")
        print(expected_form)
        print("produced F:")
        print(my_mobile.F)
        assert my_mobile.F.equals(expected_form)

    def test_with_trap_conglo_transient(self):
        """Check for the case of transient simulation with a trap conglomerate"""
        # build
        Index._globalcount = 8
        my_mobile = festim.Mobile()
        my_mobile.F = 0
        my_mobile.solution = f.Function(self.V, name="c_m")
        my_mobile.previous_solution = f.Function(self.V, name="c_m_n")
        my_mobile.test_function = f.TestFunction(self.V)
        my_mats = festim.Materials([self.mat1, self.mat2])

        trap1 = festim.Trap(
            [1, 2], [1, 2], [1, 2], [1, 2], [self.mat1, self.mat2], [1, 2]
        )
        add_functions(trap1, self.V, id=1)

        my_traps = festim.Traps([trap1])

        # run
        my_mobile.create_diffusion_form(
            my_mats, self.my_mesh, self.my_temp, dt=self.dt, traps=my_traps
        )

        # test
        Index._globalcount = 8
        v = my_mobile.test_function
        D1 = self.mat1.D_0 * f.exp(-self.mat1.E_D / festim.k_B / self.my_temp.T)
        D2 = self.mat2.D_0 * f.exp(-self.mat2.E_D / festim.k_B / self.my_temp.T)
        c_0 = my_mobile.solution
        c_0_n = my_mobile.previous_solution
        expected_form = ((c_0 - c_0_n) / self.dt.value) * v * self.my_mesh.dx(1)
        expected_form += ((c_0 - c_0_n) / self.dt.value) * v * self.my_mesh.dx(2)
        expected_form += f.dot(D1 * f.grad(c_0), f.grad(v)) * self.my_mesh.dx(1)
        expected_form += f.dot(D2 * f.grad(c_0), f.grad(v)) * self.my_mesh.dx(2)
        form_trapping_expected = 0
        for i in range(2):
            form_trapping_expected += (
                (
                    -trap1.k_0[i]
                    * f.exp(-trap1.E_k[i] / festim.k_B / self.my_temp.T)
                    * c_0
                    * (trap1.density[i] - trap1.solution)
                )
                * v
                * self.my_mesh.dx(i + 1)
            )
            form_trapping_expected += (
                trap1.p_0[i]
                * f.exp(-trap1.E_p[i] / festim.k_B / self.my_temp.T)
                * trap1.solution
                * v
                * self.my_mesh.dx(i + 1)
            )
        expected_form += -form_trapping_expected
        print("expected F:")
        print(expected_form)
        print("produced F:")
        print(my_mobile.F)
        assert my_mobile.F.equals(expected_form)


class TestInitialise:
    """
    Tests the initialise method of the festim.Mobile class
    """

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    u = f.Function(V)

    def test_from_expresion(self):
        my_mobile = festim.Mobile()
        my_mobile.previous_solution = self.u
        value = 1 + festim.x
        expected_sol = my_mobile.get_comp(self.V, value)
        my_mobile.initialise(self.V, value)

        # test
        for x in [0, 0.5, 0.3, 0.6]:
            assert my_mobile.previous_solution(x) == expected_sol(x)


def test_fluxes():
    """
    Tests the create_fluxes_form method of the festim.Mobile class
    """
    Kr_0 = 2
    E_Kr = 3
    order = 2
    k_B = festim.k_B

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mesh = festim.Mesh(mesh)
    my_mesh.dx = f.dx()
    my_mesh.ds = f.ds()

    my_mobile = festim.Mobile()
    my_mobile.F = 0
    my_mobile.solution = f.Function(V)
    my_mobile.test_function = f.TestFunction(V)
    my_mobile.boundary_conditions = [
        festim.RecombinationFlux(Kr_0=Kr_0, E_Kr=E_Kr, order=order, surfaces=1),
        festim.FluxBC(value=2 * festim.x + festim.t, surfaces=[1, 2], field=0),
    ]
    T = festim.Temperature(value=1000)
    T.create_functions(my_mesh)

    my_mobile.create_fluxes_form(T, my_mesh.ds)

    test_sol = my_mobile.test_function
    sol = my_mobile.solution
    Kr_0 = my_mobile.sub_expressions[0]
    E_Kr = my_mobile.sub_expressions[1]
    Kr = Kr_0 * f.exp(-E_Kr / k_B / T.T)
    expected_form = 0
    expected_form += -test_sol * (-Kr * (sol) ** order) * f.ds(1)
    expected_form += -test_sol * my_mobile.sub_expressions[2] * f.ds(1)
    expected_form += -test_sol * my_mobile.sub_expressions[2] * f.ds(2)
    assert expected_form.equals(my_mobile.F)
    assert expected_form.equals(my_mobile.F_fluxes)
