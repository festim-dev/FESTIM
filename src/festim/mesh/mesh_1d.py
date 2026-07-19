from mpi4py import MPI

import basix.ufl
import dolfinx.mesh
import numpy as np
import ufl

from festim.mesh.mesh import Mesh


def _is_nested(vertices) -> bool:
    """Returns True if vertices is a list of lists of coordinates."""
    return len(vertices) > 0 and all(np.ndim(block) > 0 for block in vertices)


class Mesh1D(Mesh):
    """1D Mesh.

    The vertices can be given as a flat list of x-coordinates, or as a list of
    lists of x-coordinates. In the latter case each sublist produces a
    disconnected block of cells, so that no cell is created between the last
    vertex of a block and the first vertex of the next one. This makes it
    possible to represent several solids separated by a gap (which can then be
    coupled through e.g. an enclosure).

    Args:
        vertices (list or np.ndarray): the mesh x-coordinates (m), either flat
            or as a list of lists (one per disconnected block)

    Attributes:
        vertices (np.ndarray): all the mesh x-coordinates (m), sorted
        vertex_blocks (list of np.ndarray): the x-coordinates of each
            disconnected block of the mesh

    Examples:

        .. testcode::

            import festim as F

            # a single continuous domain
            F.Mesh1D(vertices=[0, 0.1, 0.2, 0.3])

            # two blocks separated by a gap between x=0.3 and x=1
            F.Mesh1D(vertices=[[0, 0.1, 0.2, 0.3], [1, 1.1, 1.2]])
    """

    def __init__(self, vertices, **kwargs) -> None:
        self.vertices = vertices

        mesh = self.generate_mesh()
        super().__init__(mesh=mesh, **kwargs)

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        blocks = value if _is_nested(value) else [value]

        self._vertex_blocks = [
            np.sort(np.unique(block)).astype(np.float64) for block in blocks
        ]
        for block in self._vertex_blocks:
            if block.size < 2:
                raise ValueError("Each block of vertices must have at least 2 vertices")

        # sort the blocks by their first vertex so that they are ordered along x
        self._vertex_blocks.sort(key=lambda block: block[0])

        for block, next_block in zip(self._vertex_blocks[:-1], self._vertex_blocks[1:]):
            if next_block[0] < block[-1]:
                raise ValueError("Blocks of vertices must not overlap")

        self._vertices = np.concatenate(self._vertex_blocks)

    @property
    def vertex_blocks(self):
        return self._vertex_blocks

    def generate_mesh(self):
        """Generates a 1D mesh."""

        if MPI.COMM_WORLD.rank == 0:
            mesh_points = np.reshape(self.vertices, (len(self.vertices), 1))
            # cells are created within each block only, leaving the blocks
            # disconnected from each other
            cells_per_block = []
            offset = 0
            for block in self.vertex_blocks:
                indexes = np.arange(offset, offset + block.shape[0])
                cells_per_block.append(np.stack((indexes[:-1], indexes[1:]), axis=-1))
                offset += block.shape[0]
            cells = np.concatenate(cells_per_block)

        else:
            mesh_points = np.empty((0, 1), dtype=np.float64)
            cells = np.empty((0, 2), dtype=np.int64)

        degree = 1
        domain = ufl.Mesh(
            basix.ufl.element(basix.ElementFamily.P, "interval", degree, shape=(1,))
        )
        return dolfinx.mesh.create_mesh(
            comm=MPI.COMM_WORLD, cells=cells, x=mesh_points, e=domain
        )

    def check_borders(self, volume_subdomains):
        """Checks that the borders of the subdomain are within the domain.

        Args:
            volume_subdomains (list of festim.VolumeSubdomain1D): the volume subdomains

        Raises:
            Value error: if borders outside the domain
        """
        # check volume subdomain is defined
        # TODO this possible by default
        if len(volume_subdomains) == 0:
            raise ValueError("No volume subdomains defined")

        # each block of the mesh must be tiled by the subdomains that lie in it
        remaining = list(volume_subdomains)
        for block in self.vertex_blocks:
            in_block = [
                vol
                for vol in remaining
                if block[0] <= min(vol.borders) and max(vol.borders) <= block[-1]
            ]
            for vol in in_block:
                remaining.remove(vol)

            block_borders = np.sort(
                [border for vol in in_block for border in vol.borders]
            )
            if len(block_borders) == 0:
                raise ValueError("borders dont match domain borders")

            # check that subdomains are connected
            for start, end in zip(block_borders[1:-2:2], block_borders[2:-1:2]):
                if start != end:
                    raise ValueError("Subdomain borders don't match to each other")

            # check that subdomains span the whole block
            if block_borders[0] != block[0] or block_borders[-1] != block[-1]:
                raise ValueError("borders dont match domain borders")

        # subdomains that don't fit within a single block
        if remaining:
            raise ValueError("borders dont match domain borders")

    def define_meshtags(self, surface_subdomains, volume_subdomains, interfaces=None):
        # check if all borders are defined
        self.check_borders(volume_subdomains)
        return super().define_meshtags(
            surface_subdomains, volume_subdomains, interfaces
        )
