import FESTIM
import fenics as f
from ufl.core.multiindex import Index
import pytest


class TestInitialise:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    u = f.Function(V)
    vm = f.MeshFunction("size_t", mesh, 1, 1)
    T = FESTIM.Temperature(10)
    T.create_functions(FESTIM.Mesh(mesh))

    def test_from_expresion_chemical_pot(self):
        my_theta = FESTIM.Theta()
        S = f.interpolate(f.Constant(2), self.V)
        my_theta.materials = FESTIM.Materials([FESTIM.Material(1, 1, 0, S_0=2, E_S=0)])
        my_theta.volume_markers = self.vm
        my_theta.T = self.T
        my_theta.S = S
        my_theta.previous_solution = self.u
        value = 1 + FESTIM.x
        expected_sol = my_theta.get_comp(self.V, value)
        expected_sol = f.project(expected_sol / S)

        # run
        my_theta.initialise(self.V, value)

        # test
        assert f.errornorm(my_theta.previous_solution, expected_sol) == pytest.approx(0)


class TestCreateDiffusionForm:
    mesh = f.UnitIntervalMesh(10)
    my_mesh = FESTIM.Mesh(mesh)
    my_temp = FESTIM.Temperature(value=100)
    my_temp.create_functions(my_mesh)
    my_mesh.dx = f.dx()
    dt = FESTIM.Stepsize(initial_value=1)
    V = f.FunctionSpace(my_mesh.mesh, "CG", 1)
    mat1 = FESTIM.Material(1, D_0=1, E_D=1, S_0=2, E_S=3)
    mat2 = FESTIM.Material(2, D_0=2, E_D=2, S_0=3, E_S=4)

    def test_chemical_potential(self):
        # build
        Index._globalcount = 8
        my_theta = FESTIM.Theta()
        my_theta.F = 0
        my_theta.solution = f.Function(self.V, name="c_t")
        my_theta.previous_solution = f.Function(self.V, name="c_t_n")
        my_theta.test_function = f.TestFunction(self.V)
        my_mats = FESTIM.Materials([self.mat1])

        # run
        my_theta.create_diffusion_form(my_mats, self.my_mesh, self.my_temp, dt=self.dt)

        # test
        Index._globalcount = 8
        v = my_theta.test_function
        D = self.mat1.D_0 * f.exp(-self.mat1.E_D / FESTIM.k_B / self.my_temp.T)
        S = self.mat1.S_0 * f.exp(-self.mat1.E_S / FESTIM.k_B / self.my_temp.T)
        S_n = self.mat1.S_0 * f.exp(-self.mat1.E_S / FESTIM.k_B / self.my_temp.T_n)
        c_0 = my_theta.solution * S
        c_0_n = my_theta.previous_solution * S_n
        expected_form = ((c_0 - c_0_n) / self.dt.value) * v * self.my_mesh.dx(1)
        expected_form += f.dot(D * f.grad(c_0), f.grad(v)) * self.my_mesh.dx(1)

        print("expected F:")
        print(expected_form)
        print("produced F:")
        print(my_theta.F)
        assert my_theta.F.equals(expected_form)


def test_get_concentration_for_a_given_material():
    # build
    S_0 = 2
    E_S = 0.5
    my_mat = FESTIM.Material(1, 1, 1, S_0=S_0, E_S=E_S)
    my_theta = FESTIM.Theta()
    my_mesh = FESTIM.MeshFromRefinements(10, 1)
    V = f.FunctionSpace(my_mesh.mesh, "CG", 1)
    my_theta.solution = f.interpolate(f.Constant(100), V)
    my_theta.previous_solution = f.interpolate(f.Constant(200), V)

    my_temp = FESTIM.Temperature()
    my_temp.T = f.interpolate(f.Constant(300), V)
    my_temp.T_n = f.interpolate(f.Constant(500), V)

    # run
    c, c_n = my_theta.get_concentration_for_a_given_material(my_mat, my_temp)
    c = f.project(c, V)
    c_n = f.project(c_n, V)

    # test
    expected_c = f.project(
        my_theta.solution * S_0 * f.exp(-E_S / FESTIM.k_B / my_temp.T), V
    )
    expected_c_n = f.project(
        my_theta.previous_solution * S_0 * f.exp(-E_S / FESTIM.k_B / my_temp.T_n), V
    )
    assert f.errornorm(c, expected_c) == pytest.approx(0)
    assert f.errornorm(c_n, expected_c_n) == pytest.approx(0)


# TODO try to replace
# def test_mobile_concentration():
#     my_theta = FESTIM.Theta()
#     my_theta.S = 3
#     my_theta.solution = 12

#     assert my_theta.mobile_concentration() == 3*12


def test_post_processing_solution_to_concentration():
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "CG", 1)
    S = 3
    value_theta = 5
    materials = FESTIM.Materials([FESTIM.Material(1, 1, 0, S, E_S=0)])
    vm = f.MeshFunction("size_t", mesh, 1, 1)
    dx = f.Measure("dx", domain=mesh, subdomain_data=vm)
    my_theta = FESTIM.Theta()
    my_theta.S = S
    my_theta.solution = f.interpolate(f.Constant(value_theta), V)

    expected_concentration = f.project(f.Constant(S) * value_theta, V)
    my_theta.create_form_post_processing(V, materials, dx)
    my_theta.post_processing_solution_to_concentration()

    assert f.errornorm(
        my_theta.post_processing_solution, expected_concentration
    ) == pytest.approx(0)
