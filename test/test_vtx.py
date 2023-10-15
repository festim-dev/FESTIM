import festim as F
import dolfinx
from mpi4py import MPI
import numpy as np

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))


def test_vtx_export_one_function(tmpdir):
    """Test can add one function to a vtx export"""
    u = dolfinx.fem.Function(V)

    filename = tmpdir.join("my_export.bp")
    my_export = F.VTXExport(filename, field=None)
    my_export.define_writer(mesh.comm, [u])

    for t in range(10):
        u.interpolate(lambda x: t * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2))

        my_export.write(t)


def test_vtx_export_two_functions(tmpdir):
    """Test can add two functions to a vtx export"""
    u = dolfinx.fem.Function(V)
    v = dolfinx.fem.Function(V)

    filename = tmpdir.join("my_export.bp")
    my_export = F.VTXExport(filename, field=None)

    my_export.define_writer(mesh.comm, [u, v])

    for t in range(10):
        u.interpolate(lambda x: t * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2))
        v.interpolate(lambda x: t * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2))

        my_export.write(t)


def test_vtx_integration_with_h_transport_problem(tmpdir):
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    my_mat = F.Material(D_0=1, E_D=0, name="mat")
    my_model.subdomains = [
        F.VolumeSubdomain1D(1, borders=[0.0, 4.0], material=my_mat),
        F.SurfaceSubdomain1D(1, x=0.0),
        F.SurfaceSubdomain1D(2, x=4.0),
    ]
    my_model.species = [F.Species("H")]
    my_model.temperature = 500

    filename = tmpdir.join("my_export.bp")
    my_export = F.VTXExport(filename, field=my_model.species[0])
    my_model.exports = [my_export]

    my_model.initialise()

    for t in range(10):
        my_export.write(t)
