from enum import Enum

import dolfinx
import numpy as np
import ufl
from dolfinx.mesh import Mesh as dolfinx_Mesh
from dolfinx.mesh import meshtags

__all__ = ["CoordinateSystem", "Mesh"]


class CoordinateSystem(Enum):
    CARTESIAN = 10
    CYLINDRICAL = 20
    SPHERICAL = 30

    @classmethod
    def from_string(cls, s: str):
        """Can be removed with Python 3.11+."""
        s = s.lower()
        if s == "cartesian":
            return cls.CARTESIAN
        elif s == "cylindrical":
            return cls.CYLINDRICAL
        elif s == "spherical":
            return cls.SPHERICAL
        else:
            raise ValueError(
                "coordinate_system must be one of 'cartesian', 'cylindrical', or 'spherical'"  # noqa: E501
            )

    def __str__(self):
        if self == CoordinateSystem.CARTESIAN:
            return "cartesian"
        elif self == CoordinateSystem.CYLINDRICAL:
            return "cylindrical"
        elif self == CoordinateSystem.SPHERICAL:
            return "spherical"
        else:
            raise ValueError("Invalid CoordinateSystem value")


class Mesh:
    """Mesh class.

    Args:
        mesh: The mesh. Defaults to None.
        coordinate_system: the coordinate system of the mesh ("cartesian",
            "cylindrical", "spherical"). Defaults to "cartesian".

    Attributes:
        mesh The mesh
        vdim: the dimension of the mesh cells
        fdim: the dimension of the mesh facets
        n: Symbolic representation of the vector normal to the facets
            of the mesh.
    """

    mesh: dolfinx.mesh.Mesh
    _coordinate_system: CoordinateSystem

    vdim: int
    fdim: int
    n: ufl.FacetNormal

    def __init__(
        self,
        mesh: dolfinx_Mesh | None = None,
        coordinate_system: str | CoordinateSystem = CoordinateSystem.CARTESIAN,
    ):
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
            self.coordinate_system = coordinate_system

        self.check_mesh_dim_coords()

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
    def coordinate_system(self):
        return self._coordinate_system

    @coordinate_system.setter
    def coordinate_system(self, value):
        if isinstance(value, CoordinateSystem):
            self._coordinate_system = value
        elif isinstance(value, str):
            self._coordinate_system = CoordinateSystem.from_string(value)
        else:
            raise ValueError(
                "coordinate_system must be of type str or CoordinateSystem"
            )

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
        """Defines the facet and volume meshtags of the mesh.

        Args:
            surface_subdomains (list of festim.SufaceSubdomains): the surface
            subdomains of the model
            volume_subdomains (list of festim.VolumeSubdomains): the volume
            subdomains of the model
            interfaces (dict, optional): the interfaces between volume
                subdomains {int: [VolumeSubdomain, VolumeSubdomain]}. Defaults to None.

        Returns:
            dolfinx.mesh.MeshTags: the facet meshtags
            dolfinx.mesh.MeshTags: the volume meshtags
        """
        # find all cells in domain and mark them as 0
        # ghosts are included: in parallel the locators below return indices of
        # both owned and ghost entities
        cell_map = self._mesh.topology.index_map(self.vdim)
        num_cells = cell_map.size_local + cell_map.num_ghosts
        mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
        tags_volumes = np.full(num_cells, 0, dtype=np.int32)

        # find all facets in domain and mark them as 0
        facet_map = self._mesh.topology.index_map(self.fdim)
        num_facets = facet_map.size_local + facet_map.num_ghosts
        mesh_facet_indices = np.arange(num_facets, dtype=np.int32)
        tags_facets = np.full(num_facets, 0, dtype=np.int32)

        # a codim-2 surface subdomain bounds a manifold, not the mesh: its entities are
        # located on the manifold's submesh when the boundary condition using them is
        # created, and it has no facet of the parent mesh to tag. Locating it here would
        # tag whatever facets its locator happens to match, which are not its entities
        facet_surface_subdomains = [
            s for s in surface_subdomains if s.codim(self.vdim) == 1
        ]

        for surf in facet_surface_subdomains:
            try:
                # find all facets in subdomain and mark them as surf.id
                entities = surf.locate_boundary_facet_indices(self._mesh)
            except ValueError:
                if len(facet_surface_subdomains) > 1:
                    raise ValueError(
                        "Surface subdomain must have a locator attribute if"
                        " several subdomains are defined"
                    )
                self.mesh.topology.create_connectivity(self.fdim, self.fdim + 1)
                entities = dolfinx.mesh.exterior_facet_indices(self.mesh.topology)

            tags_facets[entities] = surf.id

        # codim-1 volume subdomains are manifolds embedded in the mesh: they carry a
        # transport equation like any other volume subdomain, but their entities are
        # facets, so they are tagged in the facet meshtags alongside the surfaces
        bulk_subdomains = [v for v in volume_subdomains if v.codim(self.vdim) == 0]
        manifold_subdomains = [v for v in volume_subdomains if v.codim(self.vdim) == 1]

        manifold_entities = {}
        for vol in manifold_subdomains:
            entities = vol.locate_subdomain_entities(self._mesh)
            manifold_entities[vol] = entities
            # NOTE: a manifold lying on the exterior boundary shares its facets with
            # any surface subdomain covering the same location. Tagging is
            # last-writer-wins and the manifold comes second, so the surface loses
            # those facets and every ds() on it integrates to zero
            tags_facets[entities] = vol.id

        for vol in bulk_subdomains:
            try:
                # find all cells in subdomain and mark them as vol.id
                entities = vol.locate_subdomain_entities(self._mesh)
                tags_volumes[entities] = vol.id
            except ValueError:
                if len(bulk_subdomains) > 1:
                    raise ValueError(
                        "Volume subdomain must have a locator if"
                        " several subdomains are defined"
                    )
                tags_volumes[:] = vol.id

        # make sure there are no zeros in the tags_volumes
        assert np.all(tags_volumes != 0), (
            "All cells must be tagged with a non-zero value"
            "This error can be caused by a volume subdomain not being defined"
            "correctly, or by a mesh that is too coarse to capture the geometry"
            "of the volume subdomains"
        )

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

            # an Interface and a codim-1 volume subdomain both describe what happens
            # across the same facets, in mutually exclusive ways: a jump in
            # concentration, or a transport equation on the manifold. Tagging is
            # last-writer-wins, so without this the interface would silently steal the
            # manifold's facets
            for vol, entities in manifold_entities.items():
                overlap = np.intersect1d(interface_entities, entities)
                if overlap.size:
                    raise ValueError(
                        f"interface {interface.id} and codim-1 volume subdomain "
                        f"{vol.id} share {overlap.size} facets. A pair of volume "
                        "subdomains may either have an interface condition or be "
                        "separated by a codim-1 subdomain, not both."
                    )

            tags_facets[interface_entities] = interface.id

        facet_meshtags = meshtags(
            self._mesh, self.fdim, mesh_facet_indices, tags_facets
        )

        return facet_meshtags, volume_meshtags

    def check_mesh_dim_coords(self):
        """Checks if the used coordinates can be applied for geometry with the specified
        dimensions."""

        if self.coordinate_system == CoordinateSystem.SPHERICAL and self.vdim != 1:
            raise AttributeError(
                "spherical coordinates can be used for one-dimensional domains only"
            )
        if self.coordinate_system == CoordinateSystem.CYLINDRICAL and self.vdim == 3:
            raise AttributeError(
                "cylindrical coordinates cannot be used for 3D domains"
            )
