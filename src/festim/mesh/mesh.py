import dolfinx
import numpy as np
import ufl
from dolfinx.mesh import Mesh as dolfinx_Mesh
from dolfinx.mesh import meshtags


class Mesh:
    """
    Mesh class

    Args:
        mesh The mesh. Defaults to None.

    Attributes:
        mesh The mesh
        vdim: the dimension of the mesh cells
        fdim: the dimension of the mesh facets
        n: Symbolic representation of the vector normal to the facets
            of the mesh.
    """

    _mesh: dolfinx.mesh.Mesh

    def __init__(self, mesh: dolfinx_Mesh | None = None):
        self.mesh = mesh
        if self._mesh is not None:
            # create cell to facet connectivity
            self._mesh.topology.create_connectivity(
                self._mesh.topology.dim, self._mesh.topology.dim - 1
            )

            # create facet to cell connectivity
            self._mesh.topology.create_connectivity(
                self._mesh.topology.dim - 1, self._mesh.topology.dim
            )

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if isinstance(value, dolfinx.mesh.Mesh):
            self._mesh = value
        else:
            raise TypeError("Mesh must be of type dolfinx.mesh.Mesh")

    @property
    def vdim(self):
        if self._mesh is None:
            raise RuntimeError("Mesh is not defined")
        return self._mesh.topology.dim

    @property
    def fdim(self):
        if self._mesh is None:
            raise RuntimeError("Mesh is not defined")
        return self._mesh.topology.dim - 1

    @property
    def n(self):
        if self._mesh is None:
            raise RuntimeError("Mesh is not defined")
        return ufl.FacetNormal(self._mesh)

    def define_meshtags(self, surface_subdomains, volume_subdomains, interfaces=None):
        """Defines the facet and volume meshtags of the mesh

        Args:
            surface_subdomains (list of festim.SufaceSubdomains): the surface subdomains of the model
            volume_subdomains (list of festim.VolumeSubdomains): the volume subdomains of the model
            interfaces (dict, optional): the interfaces between volume
                subdomains {int: [VolumeSubdomain, VolumeSubdomain]}. Defaults to None.

        Returns:
            dolfinx.mesh.MeshTags: the facet meshtags
            dolfinx.mesh.MeshTags: the volume meshtags
        """
        # find all cells in domain and mark them as 0
        num_cells = self._mesh.topology.index_map(self.vdim).size_local
        mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
        tags_volumes = np.full(num_cells, 0, dtype=np.int32)

        # find all facets in domain and mark them as 0
        num_facets = self._mesh.topology.index_map(self.fdim).size_local
        mesh_facet_indices = np.arange(num_facets, dtype=np.int32)
        tags_facets = np.full(num_facets, 0, dtype=np.int32)

        for surf in surface_subdomains:
            try:
                # find all facets in subdomain and mark them as surf.id
                entities = surf.locate_boundary_facet_indices(self._mesh)
                tags_facets[entities] = surf.id
            except AttributeError:
                if len(surface_subdomains) > 1:
                    raise AttributeError(
                        "Surface subdomain must have a locate_boundary_facet_indices method if"
                        " several subdomains are defined"
                    )
                self.mesh.topology.create_connectivity(self.fdim, self.fdim + 1)
                rentities = dolfinx.mesh.exterior_facet_indices(self.mesh.topology)

                tags_facets[rentities] = surf.id

        for vol in volume_subdomains:
            try:
                # find all cells in subdomain and mark them as vol.id
                entities = vol.locate_subdomain_entities(self._mesh)
                tags_volumes[entities] = vol.id
            except AttributeError:
                if len(volume_subdomains) > 1:
                    raise AttributeError(
                        "Volume subdomain must have a locate_subdomain_entities method if"
                        " several subdomains are defined"
                    )
                tags_volumes[:] = vol.id

        volume_meshtags = meshtags(
            self._mesh, self.vdim, mesh_cell_indices, tags_volumes
        )

        # tag interfaces
        interfaces = interfaces or {}  # if interfaces is None, set it to empty dict
        for interface in interfaces:
            (domain_0, domain_1) = interface.subdomains
            all_0_facets = dolfinx.mesh.compute_incident_entities(
                self._mesh.topology,
                volume_meshtags.find(domain_0.id),
                self.vdim,
                self.fdim,
            )
            all_1_facets = dolfinx.mesh.compute_incident_entities(
                self._mesh.topology,
                volume_meshtags.find(domain_1.id),
                self.vdim,
                self.fdim,
            )
            interface_entities = np.intersect1d(all_0_facets, all_1_facets)
            tags_facets[interface_entities] = interface.id

        facet_meshtags = meshtags(
            self._mesh, self.fdim, mesh_facet_indices, tags_facets
        )

        return facet_meshtags, volume_meshtags
