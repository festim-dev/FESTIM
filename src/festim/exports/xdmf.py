import warnings
from pathlib import Path

from festim.species import Species
from festim.subdomain.volume_subdomain import VolumeSubdomain

from .field import SpeciesExport


class XDMFExport(SpeciesExport):
    """Export species fields to an XDMF file.

    Thin wrapper over :class:`festim.SpeciesExport` with ``format="xdmf"``.

    .. deprecated::
        Use :class:`festim.SpeciesExport` with ``format="xdmf"`` instead. This class
        will be removed in a future release.

    Args:
        filename: The name of the output file
        field: The field(s) to export
        subdomain: The subdomain to export on. If `None` we export on all domains.
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps. Defaults to None.
    """

    def __init__(
        self,
        filename: str | Path,
        field: list[Species] | Species,
        subdomain: VolumeSubdomain = None,
        times: list[float] | list[int] | None = None,
    ) -> None:
        warnings.warn(
            "XDMFExport is deprecated and will be removed in a future release, use "
            "SpeciesExport(..., format='xdmf') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            filename, field, subdomain=subdomain, format="xdmf", times=times
        )
