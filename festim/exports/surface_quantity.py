import festim as F


class SurfaceQuantity:
    """Export SurfaceQuantity

    Args:

    Attributes:
    """

    def __init__(
        self, field, surface_subdomain, volume_subdomain, filename: str = None
    ) -> None:
        self.field = field
        self.subdomain = surface_subdomain
        self.volume_subdomain = volume_subdomain
        self.filename = filename
        self.ds = None
        self.n = None
        self.D = None
        self.S = None
