from collections.abc import Sequence

import numpy as np
import scifem
from dolfinx import fem


def evaluate_at_point(u: fem.Function, point: float | Sequence[float]):
    """Evaluate a finite element function at a physical point.

    Thin wrapper around :func:`scifem.evaluate_function`. Evaluating by
    coordinate rather than indexing ``u.x.array`` by position makes the result
    independent of DOF ordering (which changes between DOLFINx versions) and
    works for any element family and degree (Lagrange, DG, ...). ``scifem``
    handles locating the cell and, in parallel, gathering the value from the
    rank that owns it.

    Args:
        u: the function to evaluate.
        point: the coordinate, as a float (1D) or a sequence of ``gdim`` floats.

    Returns:
        The value of ``u`` at ``point``: a float for a scalar function, otherwise
        an array of its components.
    """
    gdim = u.function_space.mesh.geometry.dim
    coords = np.asarray(point, dtype=np.float64).reshape(1, gdim)
    values = np.ravel(scifem.evaluate_function(u, coords))
    return values[0] if values.size == 1 else values
