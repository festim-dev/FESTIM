from typing import Callable
import ufl
from scifem import assemble_scalar
from dolfinx import fem

from festim.exports.derived_quantity import DerivedQuantity
from festim.subdomain.surface_subdomain import SurfaceSubdomain
from festim.subdomain.volume_subdomain import VolumeSubdomain


class CustomQuantity(DerivedQuantity):
    """Export CustomQuantity.

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
    """

    def __init__(
        self,
        expr: Callable,
        subdomain: SurfaceSubdomain | VolumeSubdomain | int,
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
