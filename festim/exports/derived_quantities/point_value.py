from festim import DerivedQuantity


class PointValue(DerivedQuantity):
    """DerivedQuantity relative to a point

    Args:
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        point (int, float, tuple, list): the point coordinates
    """

    def __init__(self, field: str or int, x: int or float or tuple or list) -> None:
        super().__init__(field)
        # make sure x is an iterable
        if not hasattr(x, "__iter__"):
            x = [x]
        self.x = x
        self.title = "{} value at {}".format(field, x)

    def compute(self):
        """The value at the point"""
        return self.function(self.x)
