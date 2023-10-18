import festim as F
from dolfinx import fem, mesh
import mpi4py.MPI as MPI
import os
import numpy as np


def test_init():
    """Tests the initialisation of XDMFExport"""
    species = F.Species("H")
    my_export = F.XDMFExport(filename="my_export.xdmf", field=species)

    assert my_export.filename == "my_export.xdmf"
    assert my_export.field == [species]


def test_write(tmp_path):
    """Tests the write method of XDMFExport creates a file"""
    species = F.Species("H")
    filename = os.path.join(tmp_path, "test.xdmf")
    my_export = F.XDMFExport(filename=filename, field=species)

    my_export.define_writer(MPI.COMM_WORLD)

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = fem.FunctionSpace(domain, ("Lagrange", 1))
    u = fem.Function(V)

    species.solution = u

    for t in [0, 1, 2, 3]:
        my_export.write(t=t)

    assert os.path.exists(filename)


def test_integration_with_HTransportProblem(tmp_path):
    """Tests that XDMFExport can be used in conjunction with HTransportProblem"""
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1))
    my_mat = F.Material(D_0=1.9e-7, E_D=0.2, name="my_mat")
    my_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    my_model.subdomains = [my_subdomain]
    my_model.temperature = 500.0
    my_model.species = [F.Species("H")]
    filename = os.path.join(tmp_path, "test.xdmf")
    my_model.exports = [F.XDMFExport(filename=filename, field=my_model.species)]

    my_model.initialise()
    my_model.run(1)

    # checks that filename exists
    assert os.path.exists(filename)