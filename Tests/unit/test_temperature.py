import fenics
import FESTIM
from ufl.core.multiindex import Index
from pathlib import Path
import pytest
import numpy as np


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


def test_temperature_from_xdmf_create_functions(tmpdir):
    """Test for the TemperatureFromXDMF.create_functions().
    Creates a function, writes it to an XDMF file, then a TemperatureFromXDMF
    class is created from this file and the error norm between the written and
    read fuctions is computed to ensure they are the same.
    """
    # create function to be comapared
    mesh = fenics.UnitSquareMesh(10, 10)
    V = fenics.FunctionSpace(mesh, "CG", 1)
    expr = fenics.Expression("1 + x[0] + 2*x[1]", degree=2)
    T = fenics.interpolate(expr, V)
    # write function to temporary file
    T_file = tmpdir.join("T.xdmf")
    fenics.XDMFFile(str(Path(T_file))).write_checkpoint(
        T, "T", 0, fenics.XDMFFile.Encoding.HDF5, append=False
    )
    # TempFromXDMF needs a FESTIM mesh
    my_mesh = FESTIM.Mesh()
    my_mesh.mesh = mesh
    my_T = FESTIM.TemperatureFromXDMF(filename=str(Path(T_file)), label="T")
    my_T.create_functions(my_mesh)
    # evaluate error between original and read function
    error_L2 = fenics.errornorm(T, my_T.T, "L2")
    assert error_L2 < 1e-9


def test_temperature_from_xdmf_label_checker(tmpdir):
    """Test for the label check test within the TemperatureFromXDMF class,
    ensures that a ValueError is raised when reading a file with an
    incorrect label.
    """
    # create function to be written
    mesh = fenics.UnitSquareMesh(10, 10)
    V = fenics.FunctionSpace(mesh, "CG", 1)
    expr = fenics.Expression("1 + x[0] + 2*x[1]", degree=2)
    T = fenics.interpolate(expr, V)

    T_file = tmpdir.join("T.xdmf")
    fenics.XDMFFile(str(Path(T_file))).write_checkpoint(
        T, "T", 0, fenics.XDMFFile.Encoding.HDF5, append=False
    )
    # read file with wrong label specified
    with pytest.raises(ValueError):
        FESTIM.TemperatureFromXDMF(filename=str(Path(T_file)), label="coucou")


def test_temperature_from_xdmf_transient_case(tmpdir):
    """Test that the TemperatureFromXdmf class works in a transient
    h transport case"""
    # create temperature field xdmf
    my_model = FESTIM.Simulation(log_level=20)
    my_model.mesh = FESTIM.MeshFromVertices(vertices=np.linspace(0, 1, num=100))
    my_model.materials = FESTIM.Materials([FESTIM.Material(1, 1, 1)])
    my_model.T = FESTIM.Temperature(value=300)
    my_model.settings = FESTIM.Settings(
        transient=False,
        absolute_tolerance=1e12,
        relative_tolerance=1e-08,
    )
    my_model.initialise()
    T = my_model.T.T
    T_file = tmpdir.join("T.xdmf")
    fenics.XDMFFile(str(Path(T_file))).write_checkpoint(
        T, "T", 0, fenics.XDMFFile.Encoding.HDF5, append=False
    )

    # run transient simulation with TemperatureFromXDMF class
    my_model.T = FESTIM.TemperatureFromXDMF(filename=str(Path(T_file)), label="T")
    my_model.dt = FESTIM.Stepsize(initial_value=1)
    my_model.settings.transient = True
    my_model.settings.final_time = 10
    my_model.initialise()
    my_model.run()
