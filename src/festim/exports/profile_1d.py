import numpy as np
from festim.species import Species
from festim.subdomain import VolumeSubdomain


class Profile1DExport:
    """Class to export 1D profiles of a field in a simulation."""

    x: np.ndarray
    data: list
    field: Species
    subdomain: VolumeSubdomain
    _dofs: np.ndarray
    _sort_coords: np.ndarray

    def __init__(self, field: Species, subdomain: VolumeSubdomain = None):
        self.field = field
        self.data = []
        self.x = None
        self.subdomain = subdomain

        self._dofs = None
        self._sort_coords = None
