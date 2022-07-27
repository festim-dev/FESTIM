import festim
import fenics
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
    my_sim = festim.Simulation()
    my_sim.mesh = festim.Mesh(mesh, vm, sm)
    my_sim.mesh.define_measures()
    vm_computed, sm_computed = my_sim.mesh.volume_markers, my_sim.mesh.surface_markers

    # test
    for cell in fenics.cells(mesh):
        assert vm[cell] == vm_computed[cell]
    for facet in fenics.facets(mesh):
        assert sm[facet] == sm_computed[facet]

    assert my_sim.mesh.dx is not None
    assert my_sim.mesh.ds is not None


def test_integration_mesh_from_vertices_subdomains():
    """
    Integration test for meshing and subdomain 1D
    when parsing a list of vertices
    Checks that the cells are marked correctly
    """
    points = [0, 1, 2, 5, 12, 24]

    my_model = festim.Simulation()
    my_model.materials = festim.Materials(
        [
            festim.Material(1, None, None, borders=[0, 2]),
            festim.Material(2, None, None, borders=[2, 24]),
        ]
    )
    my_model.mesh = festim.MeshFromVertices(points)
    my_model.mesh.define_measures(my_model.materials)
    produced_mesh = my_model.mesh.mesh
    vm, sm = my_model.mesh.volume_markers, my_model.mesh.surface_markers

    # Testing
    for cell in fenics.cells(produced_mesh):
        if cell.midpoint().x() < 2:
            assert vm[cell] == 1
        elif cell.midpoint().x() > 2:
            assert vm[cell] == 2
    for facet in fenics.facets(produced_mesh):
        if facet.midpoint().x() == 0:
            assert sm[facet] == 1
        if facet.midpoint().x() == max(points):
            assert sm[facet] == 2
