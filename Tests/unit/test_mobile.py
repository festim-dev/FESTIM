import FESTIM
import fenics as f
from ufl.core.multiindex import Index


def test_mobile_create_diffusion_form():
    # TODO move this
    # build
    Index._globalcount = 8
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = FESTIM.Mobile()
    my_mobile.F = 0
    my_mobile.solution = f.Function(V)
    my_mobile.previous_solution = f.Function(V)
    my_mobile.test_function = f.TestFunction(V)

    mat = FESTIM.Material(1, D_0=1, E_D=1)
    my_mats = FESTIM.Materials([mat])
    dx = f.dx()
    T = FESTIM.Temperature("expression", value=100)
    T.create_functions(V)
    # run
    my_mobile.create_diffusion_form(my_mats, dx, T)

    # test
    c_0 = my_mobile.solution
    v = my_mobile.test_function
    Index._globalcount = 8
    expected_form = f.dot(mat.D_0 * f.exp(-mat.E_D/FESTIM.k_B/T.T)*f.grad(c_0),
                          f.grad(v))*dx(1)
    assert my_mobile.F.equals(expected_form)
    assert my_mobile.F_diffusion.equals(expected_form)


def test_mobile_create_source_form_one_dict():
    # TODO move this
    # build
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = FESTIM.Mobile()
    my_mobile.F = 0
    my_mobile.test_function = f.TestFunction(V)

    dx = f.dx()
    source_term = {"value": 2 + FESTIM.x + FESTIM.t}
    # run
    my_mobile.create_source_form(dx, source_term)

    # test
    source = my_mobile.sub_expressions[0]
    v = my_mobile.test_function
    expected_form = - source*v*dx
    assert my_mobile.F.equals(expected_form)
    assert my_mobile.F_source.equals(expected_form)


def test_mobile_create_source_form_several_sources():
    # TODO move this
    # build
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = FESTIM.Mobile()
    my_mobile.F = 0
    my_mobile.test_function = f.TestFunction(V)

    dx = f.dx()
    source_term = [
        {"value": 2 + FESTIM.x + FESTIM.t, "volumes": 1},
        {"value": 1 + FESTIM.x + FESTIM.t, "volumes": [2, 3]},
    ]
    # run
    my_mobile.create_source_form(dx, source_term)

    # test
    v = my_mobile.test_function
    expected_form = - my_mobile.sub_expressions[0]*v*dx(1)
    expected_form += - my_mobile.sub_expressions[1]*v*dx(2)
    expected_form += - my_mobile.sub_expressions[1]*v*dx(3)
    assert my_mobile.F.equals(expected_form)
    assert my_mobile.F_source.equals(expected_form)


def test_mobile_create_form():
    # TODO move this
    # build
    Index._globalcount = 8
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_mobile = FESTIM.Mobile()
    my_mobile.F = 0
    my_mobile.solution = f.Function(V)
    my_mobile.previous_solution = f.Function(V)
    my_mobile.test_function = f.TestFunction(V)

    mat = FESTIM.Material(1, D_0=1, E_D=1)
    my_mats = FESTIM.Materials([mat])
    dx = f.dx()
    T = FESTIM.Temperature("expression", value=100)
    T.create_functions(V)
    source_term = {"value": 2 + FESTIM.x + FESTIM.t}

    # run
    my_mobile.create_form(my_mats, dx, T, source_term=source_term)

    # test
    c_0 = my_mobile.solution
    v = my_mobile.test_function
    Index._globalcount = 8
    expected_form = my_mobile.F_diffusion + my_mobile.F_source
    assert my_mobile.F.equals(expected_form)


class TestCreateForm:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_temp = FESTIM.Temperature("expression", value=100)
    my_temp.create_functions(V)
    dx = f.dx()
    dt = f.Constant(1)

    mat1 = FESTIM.Material(1, D_0=1, E_D=1, S_0=2, E_S=3)
    mat2 = FESTIM.Material(2, D_0=2, E_D=2, S_0=3, E_S=4)

    def test_chemical_potential(self):
        # build
        Index._globalcount = 8
        my_mobile = FESTIM.Mobile()
        my_mobile.F = 0
        my_mobile.solution = f.Function(self.V, name="c_t")
        my_mobile.previous_solution = f.Function(self.V, name="c_t_n")
        my_mobile.test_function = f.TestFunction(self.V)
        my_mats = FESTIM.Materials([self.mat1])

        # run
        my_mobile.create_diffusion_form(my_mats, self.dx, self.my_temp, dt=self.dt, chemical_pot=True)

        # test
        Index._globalcount = 8
        v = my_mobile.test_function
        D = self.mat1.D_0 * f.exp(-self.mat1.E_D/FESTIM.k_B/self.my_temp.T)
        c_0 = my_mobile.solution*self.mat1.S_0*f.exp(-self.mat1.E_S/FESTIM.k_B/self.my_temp.T)
        c_0_n = my_mobile.previous_solution*self.mat1.S_0*f.exp(-self.mat1.E_S/FESTIM.k_B/self.my_temp.T_n)
        expected_form = ((c_0-c_0_n)/self.dt)*v*self.dx(1)
        expected_form += f.dot(D*f.grad(c_0), f.grad(v))*self.dx(1)

        print("expected F:")
        print(expected_form)
        print("produced F:")
        print(my_mobile.F)
        assert my_mobile.F.equals(expected_form)
