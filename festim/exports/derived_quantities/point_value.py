from festim import DerivedQuantity


class PointValue(DerivedQuantity):
    """
    Computes the value of a field at a given point

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        x (int, float, tuple, list): the point coordinates

    Attributes:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        x (int, float, tuple, list): the point coordinates
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the field

    .. note::
        Units are in H/m3 for hydrogen concentration and K for temperature

    """

    def __init__(self, field: str or int, x: int or float or tuple or list) -> None:
        super().__init__(field=field)
        # make sure x is an iterable
        if not hasattr(x, "__iter__"):
            x = [x]
        self.x = x

    @property
    def title(self):
        quantity_title = f"{self.field} value at {self.x}"
        if self.show_units:
            if self.field == "T":
                return quantity_title + " (K)"
            else:
                return quantity_title + " (H m-3)"
        else:
            return quantity_title

    def compute(self):
        """The value at the point"""
        return self.function(self.x)
