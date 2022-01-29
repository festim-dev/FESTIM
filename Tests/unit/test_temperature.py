import fenics
import FESTIM
from ufl.core.multiindex import Index


def test_formulation_heat_transfer_2_ids_per_mat():
    '''
    Test function define_variational_problem_heat_transfers
    catching bug described in issue #305
    '''

    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, 'P', 1)

    # Run function

    mat1 = FESTIM.Material(id=[1, 2], D_0=1, E_D=0, thermal_cond=1)
    mat2 = FESTIM.Material(id=3, D_0=0.25, E_D=0, thermal_cond=1)
    my_mats = FESTIM.Materials([mat1, mat2])
    my_temp = FESTIM.HeatTransferProblem(transient=False)

    my_temp.create_functions(V, my_mats, fenics.dx, fenics.ds, dt=FESTIM.Stepsize(initial_value=2))


def test_formulation_heat_transfer():
    '''
    Test function define_variational_problem_heat_transfers
    '''

    def thermal_cond(a):
        return a**2

    Index._globalcount = 8
    u = 1 + 2*FESTIM.x**2
    dt = FESTIM.Stepsize(initial_value=2)
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, 'P', 1)

    # create mesh functions
    surface_markers = fenics.MeshFunction(
        "size_t", mesh, mesh.topology().dim()-1, 0)
    surface_markers.set_all(0)
    for f in fenics.facets(mesh):
        x0 = f.midpoint()
        if fenics.near(x0.x(), 0):
            surface_markers[f] = 1
        if fenics.near(x0.x(), 1):
            surface_markers[f] = 2
    volume_markers = fenics.MeshFunction('size_t', mesh, 1, 1)
    ds = fenics.Measure('ds', domain=mesh, subdomain_data=surface_markers)
    dx = fenics.Measure('dx', domain=mesh, subdomain_data=volume_markers)

    mat1 = FESTIM.Material(1, D_0=1, E_D=1, thermal_cond=thermal_cond, rho=5, heat_capacity=4, borders=[0, 1])
    my_mats = FESTIM.Materials([mat1])
    bc1 = FESTIM.DirichletBC(surfaces=[1], value=u, component="T")
    bc2 = FESTIM.FluxBC(surfaces=[2], value=2, component="T")

    my_temp = FESTIM.HeatTransferProblem(transient=True, initial_value=0)
    my_temp.boundary_conditions = [bc1, bc2]
    my_temp.sources = [FESTIM.Source(-4, volume=[1], field="T")]
    my_temp.create_functions(V, my_mats, dx, ds, dt=dt)

    T = my_temp.T
    T_n = my_temp.T_n
    v = my_temp.v_T

    F = my_temp.F
    expressions = my_temp.sub_expressions
    Index._globalcount = 8

    source = expressions[0]
    expected_form = 5*4*(T - T_n)/dt.value * v * dx(1) + \
        fenics.dot(thermal_cond(T)*fenics.grad(T), fenics.grad(v))*dx(1)
    expected_form += - source*v*dx(1)

    neumann_flux = expressions[1]
    expected_form += -neumann_flux * v * ds(2)
    assert expected_form.equals(F)
