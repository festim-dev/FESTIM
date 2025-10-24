import numpy as np

from festim.species import Species
from festim.subdomain import VolumeSubdomain


class Profile1DExport:
    """Class to export 1D profiles of a field in a simulation.

    Args:
        field: the species for which the profile is computed.
        subdomain: the volume subdomain to compute the profile on. If None, the profile
            is computed over the entire domain.
        times: if provided, the profile will be exported at these timesteps.
            Otherwise, exports at all timesteps. Defaults to None.

    Attributes:
        field: the species for which the profile is computed.
        subdomain: the volume subdomain to compute the profile on. If None, the profile
            is computed over the entire domain.
        times: if provided, the profile will be exported at these timesteps. Otherwise,
            exports at all timesteps.
        x: the coordinates along which the profile is computed.
        data: the computed profile data.
        t: the list of time values at which the profile is computed.
    """

    x: np.ndarray
    data: list
    field: Species
    subdomain: VolumeSubdomain | None
    times: list[float] | None
    t: list[float]
    _dofs: np.ndarray
    _sort_coords: np.ndarray

    def __init__(
        self,
        field: Species,
        subdomain: VolumeSubdomain = None,
        times: list[float] | None = None,
    ):
        self.field = field
        self.data = []
        self.t = []
        self.x = None
        self.subdomain = subdomain
        self.times = times

        self._dofs = None
        self._sort_coords = None
