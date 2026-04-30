from collections.abc import Callable

import ufl
from dolfinx import fem
from scifem import assemble_scalar

from festim.exports.derived_quantity import DerivedQuantity
from festim.subdomain.surface_subdomain import SurfaceSubdomain
from festim.subdomain.volume_subdomain import VolumeSubdomain


class CustomQuantity(DerivedQuantity):
    r"""
    Export CustomQuantity.

    Args:
        expr: function that returns a UFL expression
        subdomain: subdomain on which the quantity is evaluated
        title: title of the exported quantity
        filename: name of the file to which the quantity is exported

    Attributes:
        expr: function that returns a UFL expression
        subdomain: subdomain on which the quantity is evaluated
        title: title of the exported quantity
        filename: name of the file to which the quantity is exported
        t: list of time values
        data: list of values of the quantity

    Usage:

    .. testcode::

        import numpy as np
        import festim as F

        material = F.Material(D_0=1, E_D=0)
        volume = F.VolumeSubdomain(id=1, material=material)
        surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1))

        def total_concentration(**kwargs):
            return kwargs["A"] + kwargs["B"]

        quantity = F.CustomQuantity(
            expr=total_concentration,
            subdomain=volume,
            title="Total quantity",
        )

        surface_quantity = F.CustomQuantity(
            expr=lambda **kwargs: -kwargs["D_A"] * ufl.dot(
                ufl.grad(kwargs["A"]), kwargs["n"]
            ),
            subdomain=surface,
            title="Surface flux",
        )

    The callable passed to ``expr`` receives keyword arguments assembled by the
    problem class. Common entries are:

    ``A``, ``B``, ...
        Concentrations of the species present in the problem (here A and B).
    ``n``
        The facet normal on the selected surface subdomain.
    ``T``
        The temperature field.
    ``D_A``, ``D_B``, ...
        Species-specific diffusion coefficients.
    ``D``
        The diffusion coefficient data, either a single field for one species or
        a dictionary keyed by species name when several species are present.
    ``x``
        The spatial coordinate (x[0], x[1], x[2]).

    For a surface quantity, the returned UFL expression can represent a flux such as

    .. math::

        q = -D\,\nabla c \cdot n

    and FESTIM will assemble

    .. math::

        Q = \int_{\Gamma} q\,\mathrm{d}\Gamma

    over the selected surface subdomain.

    The expression returned by ``expr`` is treated as an integrand and assembled over
    the chosen subdomain.

    .. math::

        Q = \int_{\Omega} q\,\mathrm{d}\Omega

    where ``q`` is the UFL expression returned by ``expr`` and ``\Omega`` is either a
    surface or a volume subdomain.
    """

    def __init__(
        self,
        expr: Callable,
        subdomain: SurfaceSubdomain | VolumeSubdomain,
        title: str = "Custom Quantity",
        filename: str | None = None,
    ) -> None:
        super().__init__(filename=filename)
        self.expr = expr
        self.subdomain = subdomain
        self._title = title
        self.ufl_expr = None

    @property
    def title(self):
        return self._title

    def compute(self, measure: ufl.Measure, entity_maps: dict | None = None):
        """Computes the value of the custom quantity and appends it to the data list.

        Args:
            measure: volume or surface measure of the model
            entity_maps: entity maps relating parent mesh and submesh
        """
        if self.ufl_expr is None:
            raise ValueError("The UFL expression has not been evaluated yet.")

        form = fem.form(
            self.ufl_expr * measure(self.subdomain.id), entity_maps=entity_maps
        )
        self.value = assemble_scalar(form)
        self.data.append(self.value)
