
class Source:
    """Volumetric source term.

    Attributes:
        value (sympy.Add, float): the value of the volumetric source term
        volume (int): the volume in which the source is applied
        field (str): the field on which the source is applied ("0", "solute",
            "1", "T")
    """
    def __init__(self, value, volume, field) -> None:
        """Inits Source

        Args:
            value (sympy.Add, float): the value of the volumetric source term
            volume (int): the volume in which the source is applied
            field (str): the field on which the source is applied ("0", "solute",
                "1", "T")
        """
        self.value = value
        self.volume = volume
        self.field = field
