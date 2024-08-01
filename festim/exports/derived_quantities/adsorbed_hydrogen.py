from festim import SurfaceQuantity, k_B
import fenics as f


class AdsorbedHydrogen(SurfaceQuantity):
    """
    Object to compute the value of the adsorbed H concentration,
    defined with the ``SurfaceKinetics`` boundary condition on a given surface.

    .. warning::

        The ``AdsorbedHydrogen`` export can be used only if the ``SurfaceKinetics``
        condition is defined on the same surface!

    Args:
        surface (int): the surface id

    Attributes:
        surface (int): the surface id
        export_unit (str): the unit of the derived quantity in the export file
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities file
        function (dolfin.function.function.Function): the solution function of the hydrogen adsorbed field

    """

    def __init__(self, surface) -> None:
        super().__init__(field="adsorbed", surface=surface)

    @property
    def export_unit(self):
        return f"H m-2"

    @property
    def title(self):
        quantity_title = f"Adsorbed H on surface {self.surface}"
        if self.show_units:
            return quantity_title + f" ({self.export_unit})"
        else:
            return quantity_title

    def compute(self):
        return f.assemble(self.function * self.ds(self.surface))
