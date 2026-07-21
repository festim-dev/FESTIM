"""Pytest markers for features that depend on the installed dolfinx version."""

import dolfinx
import pytest
from packaging.version import Version

from festim.enclosure._utils import MINIMUM_DOLFINX_VERSION

_supports_real_spaces = Version(dolfinx.__version__) >= Version(MINIMUM_DOLFINX_VERSION)

requires_dolfinx_011 = pytest.mark.skipif(
    not _supports_real_spaces,
    reason=(
        f"Gas enclosures require dolfinx >= {MINIMUM_DOLFINX_VERSION} (real function "
        f"spaces), found {dolfinx.__version__}"
    ),
)

requires_dolfinx_010 = pytest.mark.skipif(
    _supports_real_spaces,
    reason=(
        f"Tests the rejection path for dolfinx < {MINIMUM_DOLFINX_VERSION}, found "
        f"{dolfinx.__version__}"
    ),
)
