import dolfinx
import numpy as np
from dolfinx.cpp.fem import compute_integration_domains
from packaging.version import Version

from festim.subdomain.volume_subdomain import VolumeSubdomain


class Interface:
    id: int
    subdomains: tuple[VolumeSubdomain, VolumeSubdomain]
    parent_mesh: dolfinx.mesh.Mesh
    mt: dolfinx.mesh.MeshTags
    restriction: list[str, str] = ("+", "-")
    padded: bool

    def __init__(
        self,
        id: int,
        subdomains: list[VolumeSubdomain],
        penalty_term: float = 10.0,
    ):
        """Class representing an interface between two subdomains.

        Args:
            id (int): the tag of the interface subdomain in the parent meshtags
            subdomains (list[F.VolumeSubdomain]): the subdomains sharing this interface
            penalty_term (float, optional): Penalty term in the Nietsche DG formulation.
                Needs to be "sufficiently large". Defaults to 10.0.
        """
        self.id = id
        self.subdomains = tuple(subdomains)
        self.penalty_term = penalty_term

    def pad_parent_maps(self):
        """Workaround to make sparsity-pattern work without skips"""

        if Version(dolfinx.__version__) == Version("0.9.0"):
            args = (
                dolfinx.fem.IntegralType.interior_facet,
                self.parent_mesh.topology._cpp_object,
                self.mt.find(self.id),
                self.mt.dim,
            )
        elif Version(dolfinx.__version__) > Version("0.9.0"):
            args = (
                dolfinx.fem.IntegralType.interior_facet,
                self.parent_mesh.topology._cpp_object,
                self.mt.find(self.id),
            )

        integration_data = compute_integration_domains(*args).reshape(-1, 4)
        for i in range(2):
            # We pad the parent to submesh map to make sure that sparsity pattern is correct
            mapped_cell_0 = self.subdomains[i].parent_to_submesh[integration_data[:, 0]]
            mapped_cell_1 = self.subdomains[i].parent_to_submesh[integration_data[:, 2]]
            max_cells = np.maximum(mapped_cell_0, mapped_cell_1)
            self.subdomains[i].parent_to_submesh[integration_data[:, 0]] = max_cells
            self.subdomains[i].parent_to_submesh[integration_data[:, 2]] = max_cells
            self.subdomains[i].padded = True

    def compute_mapped_interior_facet_data(self, mesh):
        """
        Compute integration data for interface integrals.
        We define the first domain on an interface as the "+" restriction,
        meaning that we must sort all integration entities in this order

        Returns
            integration_data: Integration data for interior facets
        """
        assert (not self.subdomains[0].padded) and (not self.subdomains[1].padded)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

        if Version(dolfinx.__version__) == Version("0.9.0"):
            args = (
                dolfinx.fem.IntegralType.interior_facet,
                self.parent_mesh.topology._cpp_object,
                self.mt.find(self.id),
                self.mt.dim,
            )
        elif Version(dolfinx.__version__) > Version("0.9.0"):
            args = (
                dolfinx.fem.IntegralType.interior_facet,
                mesh.topology._cpp_object,
                self.mt.find(self.id),
            )

        integration_data = compute_integration_domains(*args)

        ordered_integration_data = integration_data.reshape(-1, 4).copy()

        mapped_cell_0 = self.subdomains[0].parent_to_submesh[integration_data[0::4]]
        mapped_cell_1 = self.subdomains[0].parent_to_submesh[integration_data[2::4]]

        switch = mapped_cell_1 > mapped_cell_0
        # Order restriction on one side
        if True in switch:
            ordered_integration_data[switch, :] = ordered_integration_data[switch][
                :, [2, 3, 0, 1]
            ]

        # Check that other restriction lies in other interface
        domain1_cell = self.subdomains[1].parent_to_submesh[
            ordered_integration_data[:, 2]
        ]
        assert (domain1_cell >= 0).all()

        return (self.id, ordered_integration_data.reshape(-1))
