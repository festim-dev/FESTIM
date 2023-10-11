import ufl


class Mesh:
    """
    Mesh class

    Attributes:
        mesh (dolfinx.mesh.Mesh): the mesh
    """

    def __init__(self, mesh=None):
        """Inits Mesh
        Args:
            mesh (dolfinx.mesh.Mesh, optional): the mesh. Defaults to None.
        """
        self.mesh = mesh

        if self.mesh is not None:
            # create cell to facet connectivity
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim, self.mesh.topology.dim - 1
            )

            # create facet to cell connectivity
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim - 1, self.mesh.topology.dim
            )
