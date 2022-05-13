import fenics
import FESTIM
from ufl.core.multiindex import Index
from pathlib import Path


def test_formulation_heat_transfer_2_ids_per_mat():
    """
    Test function define_variational_problem_heat_transfers
    catching bug described in issue #305
    """

    my_mesh = FESTIM.Mesh(fenics.UnitIntervalMesh(10))
    my_mesh.dx = fenics.dx
    my_mesh.ds = fenics.ds
    # Run function

    mat1 = FESTIM.Material(id=[1, 2], D_0=1, E_D=0, thermal_cond=1)
    mat2 = FESTIM.Material(id=3, D_0=0.25, E_D=0, thermal_cond=1)
    my_mats = FESTIM.Materials([mat1, mat2])
    my_temp = FESTIM.HeatTransferProblem(transient=False)

    my_temp.create_functions(my_mats, my_mesh, dt=FESTIM.Stepsize(initial_value=2))


def test_formulation_heat_transfer():
    """
    Test function define_variational_problem_heat_transfers
    """

    def thermal_cond(a):
        return a**2

    Index._globalcount = 8
    u = 1 + 2 * FESTIM.x**2
    dt = FESTIM.Stepsize(initial_value=2)
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, "P", 1)

    # create mesh functions
    surface_markers = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    surface_markers.set_all(0)
    for f in fenics.facets(mesh):
        x0 = f.midpoint()
        if fenics.near(x0.x(), 0):
            surface_markers[f] = 1
        if fenics.near(x0.x(), 1):
            surface_markers[f] = 2
    volume_markers = fenics.MeshFunction("size_t", mesh, 1, 1)
    ds = fenics.Measure("ds", domain=mesh, subdomain_data=surface_markers)
    dx = fenics.Measure("dx", domain=mesh, subdomain_data=volume_markers)

    my_mesh = FESTIM.Mesh(mesh, volume_markers, surface_markers)
    my_mesh.dx = dx
    my_mesh.ds = ds

    mat1 = FESTIM.Material(
        1,
        D_0=1,
        E_D=1,
        thermal_cond=thermal_cond,
        rho=5,
        heat_capacity=4,
        borders=[0, 1],
    )
    my_mats = FESTIM.Materials([mat1])
    bc1 = FESTIM.DirichletBC(surfaces=[1], value=u, field="T")
    bc2 = FESTIM.FluxBC(surfaces=[2], value=2, field="T")

    my_temp = FESTIM.HeatTransferProblem(transient=True, initial_value=0)
    my_temp.boundary_conditions = [bc1, bc2]
    my_temp.sources = [FESTIM.Source(-4, volume=[1], field="T")]
    my_temp.create_functions(my_mats, my_mesh, dt=dt)

    T = my_temp.T
    T_n = my_temp.T_n
    v = my_temp.v_T

    F = my_temp.F
    expressions = my_temp.sub_expressions
    Index._globalcount = 8

    source = expressions[0]
    expected_form = 5 * 4 * (T - T_n) / dt.value * v * dx(1) + fenics.dot(
        thermal_cond(T) * fenics.grad(T), fenics.grad(v)
    ) * dx(1)
    expected_form += -source * v * dx(1)

    neumann_flux = expressions[1]
    expected_form += -neumann_flux * v * ds(2)
    assert expected_form.equals(F)


def test_temp_from_xdmf_create_functions(tmpdir):
    """Test for the TempFromXDMF class
    A function is created and exported to xdmf. The reads
    same mesh from the xdmf and compares the two
    """
    mesh = fenics.UnitSquareMesh(10, 10)
    V = fenics.FunctionSpace(mesh, "CG", 1)
    expr = fenics.Expression("1 + x[0] + 2*x[1]", degree=2)
    T = fenics.interpolate(expr, V)

    T_file = tmpdir.join("T.xdmf")
    fenics.XDMFFile(str(Path(T_file))).write_checkpoint(
        T, "T", 0, fenics.XDMFFile.Encoding.HDF5, append=False
    )
    my_mesh = FESTIM.Mesh()
    my_mesh.mesh = mesh
    my_T = FESTIM.TempFromXDMF(filename=str(Path(T_file)), label="T")
    my_T.create_functions(my_mesh)
    error_L2 = fenics.errornorm(T, my_T.T, "L2")
    assert error_L2 < 1e-9
