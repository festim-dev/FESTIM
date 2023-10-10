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

    def create_measures_and_tags(self, function_space):
        """Creates the ufl.measure.Measure objects for self.ds and
        self.dx, also passes the facet and volume tags
        """
        facet_tags, volume_tags = self.create_meshtags(function_space)
        dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=volume_tags)
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tags)
        return (
            facet_tags,
            volume_tags,
            dx,
            ds,
        )