from .interface import Interface
from .surface_subdomain import SurfaceSubdomain, SurfaceSubdomain1D
from .volume_subdomain import (
    VolumeSubdomain,
    VolumeSubdomain1D,
    map_surface_to_volume_subdomains,
)

__all__ = [
    "Interface",
    "Subdomain",
    "SurfaceSubdomain",
    "SurfaceSubdomain1D",
    "VolumeSubdomain",
    "VolumeSubdomain1D",
    "map_surface_to_volume_subdomains",
]
