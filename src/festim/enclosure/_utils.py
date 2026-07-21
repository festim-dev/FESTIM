import dolfinx
from packaging.version import Version

MINIMUM_DOLFINX_VERSION = "0.11"


def check_dolfinx_version_for_enclosures():
    """Checks that the installed version of dolfinx supports real function spaces.

    Enclosure pressures are unknowns living in a real function space (a single global
    degree of freedom). Real function spaces are only available from dolfinx 0.11.

    Raises:
        NotImplementedError: if the installed dolfinx is older than 0.11
    """
    if Version(dolfinx.__version__) < Version(MINIMUM_DOLFINX_VERSION):
        raise NotImplementedError(
            f"Gas enclosures require dolfinx >= {MINIMUM_DOLFINX_VERSION} (found "
            f"{dolfinx.__version__}). Enclosure pressures are solved in a real "
            "function space, which is not implemented in older versions of dolfinx. "
            "Please upgrade dolfinx, or remove the enclosures from your model."
        )
