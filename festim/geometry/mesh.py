import ufl


class Mesh:
    """
    Mesh class
    Attributes:
        mesh (dolfinx.Mesh): the mesh
        volume_markers (dolfinx.MeshTags): markers of the mesh cells
        surface_markers (dolfinx.MeshTags): markers of the mesh facets
        dx (dolfinx.Measure):
        ds (dolfinx.Measure):
    """

    def __init__(
        self,
        mesh=None,
        volume_markers=None,
        surface_markers=None,
        subdomains=[],
        **kwargs
    ):
        """Inits Mesh
        Args:
            mesh (dolfinx.Mesh, optional): the mesh. Defaults to None.
            volume_markers (dolfinx.MeshTags, optional): markers of the mesh cells. Defaults to None.
            surface_markers (dolfinx.MeshTags, optional): markers of the mesh facets. Defaults to None.
            subdomains (list, optional): list of festimx.Subdomain objects
        """
        self.mesh = mesh
        self.volume_markers = volume_markers
        self.surface_markers = surface_markers
        self.subdomains = subdomains

        self.dx = None
        self.ds = None

        if self.mesh is not None:
            # create cell to facet connectivity
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim, self.mesh.topology.dim - 1
            )

            # create facet to cell connectivity
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim - 1, self.mesh.topology.dim
            )

    def define_markers(self, function_space):
        self.surface_markers = self.define_surface_markers(function_space)

    def define_measures(self):
        """Creates the ufl.Measure objects for self.ds"""

        self.ds = ufl.Measure(
            "ds", domain=self.mesh, subdomain_data=self.surface_markers
        )
