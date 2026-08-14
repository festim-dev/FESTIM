"""Backwards-compatible names for the derived quantities.

Before the ``domain``-based quantities, each of ``Total``, ``Average``, ``Minimum`` and
``Maximum`` came in a ``*Surface`` and a ``*Volume`` flavour that differed only in the
measure and the meshtags they were handed. These shims keep the old names, the old
constructor keyword (``surface=``/``volume=``), the old attribute and the old
``compute`` signature working, and emit a ``DeprecationWarning`` on construction.

They subclass the new classes, so ``isinstance(export, F.Total)`` holds for an
``F.TotalVolume`` and the problem needs no knowledge of the old names.
"""

import warnings

from festim.exports.quantity import Average, Maximum, Minimum, Total


def _deprecation(old: str, new: str, keyword: str) -> None:
    warnings.warn(
        f"F.{old} is deprecated and will be removed in a future release. "
        f"Use F.{new}(field=..., domain=...) instead; it takes a volume or a surface "
        f"subdomain in place of `{keyword}=`.",
        DeprecationWarning,
        stacklevel=3,
    )


class _VolumeAlias:
    """Accepts ``volume=`` and exposes ``.volume``."""

    def __init__(self, field, volume, filename=None):
        _deprecation(type(self).__name__, self._new_name, "volume")
        super().__init__(field=field, domain=volume, filename=filename)

    @property
    def volume(self):
        return self.domain

    @volume.setter
    def volume(self, value):
        self.domain = value


class _SurfaceAlias:
    """Accepts ``surface=`` and exposes ``.surface``."""

    def __init__(self, field, surface, filename=None):
        _deprecation(type(self).__name__, self._new_name, "surface")
        super().__init__(field=field, domain=surface, filename=filename)

    @property
    def surface(self):
        return self.domain

    @surface.setter
    def surface(self, value):
        self.domain = value


class _LegacyIntegralCompute:
    """Keeps ``compute(u, dx=...)`` and ``compute(u, ds=...)`` working.

    ``restriction`` is accepted and forwarded rather than swallowed: the discontinuous
    problem passes it by keyword for every surface integral, so dropping it here would
    read an interior-facet quantity off an arbitrary side.
    """

    def compute(
        self, u, measure=None, entity_maps=None, *, dx=None, ds=None, restriction=None
    ):
        measure = next(m for m in (measure, dx, ds) if m is not None)
        return super().compute(
            u=u, measure=measure, entity_maps=entity_maps, restriction=restriction
        )


class _LegacyExtremumCompute:
    """Keeps the no-argument ``compute()`` working.

    The old classes were handed their meshtags by the problem as an attribute
    (``volume_meshtags`` / ``facet_meshtags``) and read the solution off the species
    themselves. Both are recovered here so that code doing that still runs.
    """

    #: name of the attribute the problem used to set
    _legacy_meshtags_attr: str
    #: codimension of the entities the old class located dofs on
    _legacy_codim: int

    def compute(self, u=None, meshtags=None, entity_dim=None):
        if u is None:
            u = self.field.post_processing_solution
            if not hasattr(u, "function_space"):
                u = self.field.sub_function_space
        if meshtags is None:
            meshtags = getattr(self, self._legacy_meshtags_attr)
        if entity_dim is None:
            entity_dim = u.function_space.mesh.topology.dim - self._legacy_codim
        return super().compute(u=u, meshtags=meshtags, entity_dim=entity_dim)


class TotalVolume(_VolumeAlias, _LegacyIntegralCompute, Total):
    _new_name = "Total"


class TotalSurface(_SurfaceAlias, _LegacyIntegralCompute, Total):
    _new_name = "Total"


class AverageVolume(_VolumeAlias, _LegacyIntegralCompute, Average):
    _new_name = "Average"


class AverageSurface(_SurfaceAlias, _LegacyIntegralCompute, Average):
    _new_name = "Average"


class MaximumVolume(_VolumeAlias, _LegacyExtremumCompute, Maximum):
    _new_name = "Maximum"
    _legacy_meshtags_attr = "volume_meshtags"
    _legacy_codim = 0


class MaximumSurface(_SurfaceAlias, _LegacyExtremumCompute, Maximum):
    _new_name = "Maximum"
    _legacy_meshtags_attr = "facet_meshtags"
    _legacy_codim = 1


class MinimumVolume(_VolumeAlias, _LegacyExtremumCompute, Minimum):
    _new_name = "Minimum"
    _legacy_meshtags_attr = "volume_meshtags"
    _legacy_codim = 0


class MinimumSurface(_SurfaceAlias, _LegacyExtremumCompute, Minimum):
    _new_name = "Minimum"
    _legacy_meshtags_attr = "facet_meshtags"
    _legacy_codim = 1
