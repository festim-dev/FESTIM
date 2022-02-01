import FESTIM
import fenics as f
from ufl.core.multiindex import Index


class TestInitialise:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    u = f.Function(V)

    def test_from_expresion_chemical_pot(self):
        my_theta = FESTIM.Theta()
        S = f.interpolate(f.Constant(2), self.V)
        my_theta.S = S
        my_theta.previous_solution = self.u
        value = 1 + FESTIM.x
        expected_sol = my_theta.get_comp(self.V, value)
        expected_sol = f.project(expected_sol/S)

        # run
        my_theta.initialise(self.V, value)

        # test
        for x in [0, 0.5, 0.3, 0.6]:
            assert my_theta.previous_solution(x) == expected_sol(x)


class TestCreateDiffusionForm:
    mesh = f.UnitIntervalMesh(10)
    my_mesh = FESTIM.Mesh(mesh)
    my_temp = FESTIM.Temperature(value=100)
    my_temp.create_functions(my_mesh)
    dx = f.dx()
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
        my_theta.create_diffusion_form(my_mats, self.dx, self.my_temp, dt=self.dt)

        # test
        Index._globalcount = 8
        v = my_theta.test_function
        D = self.mat1.D_0 * f.exp(-self.mat1.E_D/FESTIM.k_B/self.my_temp.T)
        c_0 = my_theta.solution*self.mat1.S_0*f.exp(-self.mat1.E_S/FESTIM.k_B/self.my_temp.T)
        c_0_n = my_theta.previous_solution*self.mat1.S_0*f.exp(-self.mat1.E_S/FESTIM.k_B/self.my_temp.T_n)
        expected_form = ((c_0-c_0_n)/self.dt.value)*v*self.dx(1)
        expected_form += f.dot(D*f.grad(c_0), f.grad(v))*self.dx(1)

        print("expected F:")
        print(expected_form)
        print("produced F:")
        print(my_theta.F)
        assert my_theta.F.equals(expected_form)
