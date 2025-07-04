from festim import MinimumSurface
import fenics as f
import numpy as np
import pytest
from dolfin import MPI


@pytest.mark.parametrize("field,surface", [("solute", 1), ("T", 2)])
def test_title(field, surface):
    """
    A simple test to check that the title is set
    correctly in festim.MinimumSurface

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
    """

    my_min = MinimumSurface(field, surface)
    assert my_min.title == f"Minimum {field} surface {surface} ({my_min.export_unit})"


class TestCompute:
    """Test that the minimum surface export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    surface_markers = f.MeshFunction("size_t", mesh, 0, 0)

    right = f.CompiledSubDomain("near(x[0], 1) && on_boundary")
    right.mark(surface_markers, 1)

    dx = f.Measure("dx", domain=mesh, subdomain_data=surface_markers)

    surface = 1
    my_min = MinimumSurface("solute", surface)
    my_min.function = c
    my_min.dx = dx

    def test_minimum(self):
        dm = self.V.dofmap()
        facets = self.surface_markers.where_equal(self.surface)
        entity_closure_dofs = np.array(
            dm.entity_closure_dofs(self.mesh, self.mesh.topology().dim() - 1, facets),
            dtype=np.int32,
        )
        local_dofs = entity_closure_dofs < len(self.c.vector().get_local())
        if len(local_dofs) == 0:
            local_min = np.inf
        else:
            local_min = np.min(
                self.c.vector().get_local()[entity_closure_dofs[local_dofs]]
            )

        expected = MPI.min(self.mesh.mpi_comm(), local_min)

        produced = self.my_min.compute(self.surface_markers)
        assert produced == expected
