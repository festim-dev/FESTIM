import ufl
import dolfinx


class Mesh:
    """
    Mesh class

    Args:
        mesh (dolfinx.mesh.Mesh, optional): the mesh. Defaults to None.

    Attributes:
        mesh (dolfinx.mesh.Mesh): the mesh
        vdim (int): the dimension of the mesh cells
        fdim (int): the dimension of the mesh facets
        n (ufl.FacetNormal): the normal vector to the facets
    """

    def __init__(self, mesh=None):
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

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if isinstance(value, dolfinx.mesh.Mesh):
            self._mesh = value
        else:
            raise TypeError(f"Mesh must be of type dolfinx.mesh.Mesh")

    @property
    def vdim(self):
        return self.mesh.topology.dim

    @property
    def fdim(self):
        return self.mesh.topology.dim - 1

    @property
    def n(self):
        return ufl.FacetNormal(self.mesh)
