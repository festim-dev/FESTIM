import festim as F
import dolfinx


class CustomFenicsMesh(F.Mesh):
    """
    Use cutom fenics preset mesh

    Args:
        mesh (dolfinx.mesh.Mesh): the mesh
        surface_meshtags (dolfinx.mesh.MeshTags): surface meshtags of the mesh
        volume_meshtags (dolfinx.mesh.MeshTags): volume meshtags of the mesh

    Attributes:
        mesh (dolfinx.mesh.Mesh): the mesh
        surface_meshtags (dolfinx.mesh.MeshTags): surface meshtags of the mesh
        volume_meshtags (dolfinx.mesh.MeshTags): volume meshtags of the mesh
    """

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if value is None:
            self._mesh = value
        elif isinstance(value, dolfinx.mesh.Mesh):
            self._mesh = value
        else:
            raise TypeError(f"Mesh must be of type dolfinx.mesh.Mesh")

    @property
    def surface_meshtags(self):
        return self._surface_meshtags

    @surface_meshtags.setter
    def surface_meshtags(self, value):
        if value is None:
            self._surface_meshtags = value
        elif isinstance(value, dolfinx.mesh.MeshTags):
            self._surface_meshtags = value
        else:
            raise TypeError(f"value must be of type dolfinx.mesh.MeshTags")

    @property
    def volume_meshtags(self):
        return self._volume_meshtags

    @volume_meshtags.setter
    def volume_meshtags(self, value):
        if value is None:
            self._volume_meshtags = value
        elif isinstance(value, dolfinx.mesh.MeshTags):
            self._volume_meshtags = value
        else:
            raise TypeError(f"value must be of type dolfinx.mesh.MeshTags")

    def __init__(
        self,
        mesh,
        surface_meshtags,
        volume_meshtags,
    ):
        super().__init__(mesh=mesh)

        self.mesh = mesh
        self.surface_meshtags = surface_meshtags
        self.volume_meshtags = volume_meshtags
