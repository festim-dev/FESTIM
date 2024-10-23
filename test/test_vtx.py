from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import festim as F

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


def test_vtx_export_one_function(tmpdir):
    """Test can add one function to a vtx export"""
    u = dolfinx.fem.Function(V)
    sp = F.Species("H")
    sp.post_processing_solution = u
    filename = str(tmpdir.join("my_export.bp"))
    my_export = F.VTXSpeciesExport(filename, field=sp)

    functions = my_export.get_functions()
    assert len(functions) == 1
    assert functions[0] == u


def test_vtx_export_two_functions(tmpdir):
    """Test can add two functions to a vtx export"""
    u = dolfinx.fem.Function(V)
    v = dolfinx.fem.Function(V)

    sp1 = F.Species("1")
    sp2 = F.Species("2")
    sp1.post_processing_solution = u
    sp2.post_processing_solution = v
    filename = str(tmpdir.join("my_export.bp"))
    my_export = F.VTXSpeciesExport(filename, field=[sp1, sp2])

    functions = my_export.get_functions()
    assert len(functions) == 2
    assert functions[0] == u
    assert functions[1] == v


@pytest.mark.skip(reason="Not implemented")
def test_vtx_export_subdomain():
    """Test that given multiple subdomains in problem,
    only correct functions are extracted from species"""
    pass


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

    filename = str(tmpdir.join("my_export.bp"))
    my_export = F.VTXSpeciesExport(filename, field=my_model.species[0])
    my_model.exports = [my_export]
    my_model.settings = F.Settings(atol=1, rtol=0.1)
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    my_model.initialise()
    assert len(my_export.get_functions()) == 1
    assert len(my_model._vtxfiles) == 1


def test_field_attribute_is_always_list():
    """Test that the field attribute is always a list"""
    my_export = F.VTXSpeciesExport("my_export.bp", field=F.Species("H"))
    assert isinstance(my_export.field, list)

    my_export = F.VTXSpeciesExport("my_export.bp", field=[F.Species("H")])
    assert isinstance(my_export.field, list)


@pytest.mark.parametrize("field", [["H", 1], 1, [F.Species("H"), 1]])
def test_field_attribute_raises_error_when_invalid_type(field):
    """Test that the field attribute raises an error if the type is not festim.Species or list"""
    with pytest.raises(TypeError):
        F.VTXSpeciesExport("my_export.bp", field=field)


def test_filename_raises_error_when_wrong_type():
    """Test that the filename attribute raises an error if the extension is not .bp"""
    with pytest.raises(TypeError):
        F.VTXSpeciesExport(1, field=[F.Species("H")])
