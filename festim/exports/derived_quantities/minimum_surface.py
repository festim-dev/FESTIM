from festim import SurfaceQuantity
from dolfin import MPI
import numpy as np


class MinimumSurface(SurfaceQuantity):
    """
    Computes the minimum value of a field on a given surface

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id

    Attributes:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the field

    .. note::
        Units are in H/m3 for hydrogen concentration and K for temperature

    """

    def __init__(self, field, surface) -> None:
        super().__init__(field=field, surface=surface)

    @property
    def export_unit(self):
        if self.field == "T":
            return "K"
        else:
            return "H m-3"

    @property
    def title(self):
        quantity_title = f"Minimum {self.field} surface {self.surface}"
        if self.show_units:
            return quantity_title + f" ({self.export_unit})"
        else:
            return quantity_title

    def compute(self, surface_markers):
        """Minimum of f over subdomains facets marked with self.surface"""
        V = self.function.function_space()
        mesh = surface_markers.mesh()
        dm = V.dofmap()

        facets = surface_markers.where_equal(self.surface)
        entity_closure_dofs = np.array(
            dm.entity_closure_dofs(mesh, mesh.topology().dim() - 1, facets),
            dtype=np.int32,
        )
        local_dofs = entity_closure_dofs < len(self.function.vector().get_local())
        if len(local_dofs) == 0:
            local_min = np.inf
        else:
            local_min = np.min(
                self.function.vector().get_local()[entity_closure_dofs[local_dofs]]
            )

        global_min = MPI.max(mesh.mpi_comm(), local_min)

        return global_min
