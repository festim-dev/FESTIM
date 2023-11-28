from festim import DerivedQuantity


class PointValue(DerivedQuantity):
    """DerivedQuantity relative to a point

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        point (int, float, list): the point coordinates
    """

    def __init__(self, field: str or int, x: float or list) -> None:
        super().__init__(field)
        self.x = x
        self.title = "{} value at {}".format(field, x)

    def compute(self):
        """The value at the point"""
        return self.function(self.x)
