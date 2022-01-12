import os.path
from os import path
import FESTIM
import fenics
import pytest
from pathlib import Path


def test_define_markers(tmpdir):
    """Checks that markers can be defined from XDMF files and that the mesh
    functions have the correct values.
    """
    # build
    mesh = fenics.UnitSquareMesh(16, 16)

    vm = fenics.MeshFunction("size_t", mesh, mesh.topology().dim())
    for i, cell in enumerate(fenics.cells(mesh)):
        vm[cell] = i
    sm = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    for i, facet in enumerate(fenics.facets(mesh)):
        sm[facet] = i

    filename_volume = tmpdir.join("vm.xdmf")
    fenics.XDMFFile(str(Path(filename_volume))).write(vm)
    filename_surface = tmpdir.join("sm.xdmf")
    fenics.XDMFFile(str(Path(filename_surface))).write(sm)

    # run
    my_sim = FESTIM.Simulation(parameters={"boundary_conditions": []})
    my_sim.mesh = FESTIM.Mesh(mesh, vm, sm)
    my_sim.parameters["mesh_parameters"] = {}
    my_sim.parameters["mesh_parameters"]["mesh_file"] = \
        str(Path(filename_volume))
    my_sim.parameters["mesh_parameters"]["cells_file"] = \
        str(Path(filename_volume))
    my_sim.parameters["mesh_parameters"]["facets_file"] = \
        str(Path(filename_surface))
    my_sim.define_markers()
    vm_computed, sm_computed = my_sim.volume_markers, my_sim.surface_markers

    # test
    for cell in fenics.cells(mesh):
        assert vm[cell] == vm_computed[cell]
    for facet in fenics.facets(mesh):
        assert sm[facet] == sm_computed[facet]

    assert my_sim.dx is not None
    assert my_sim.ds is not None
