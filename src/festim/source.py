from dolfinx import fem

import festim as F


class SourceBase:
    """
    Source class

    Args:
        volume (festim.VolumeSubdomain1D): the volume subdomains where the source is applied
        value (float, int, fem.Constant or callable): the value of the soure
        species (festim.Species): the species to which the source is applied

    Attributes:
        volume (festim.VolumeSubdomain1D): the volume subdomains where the source is applied
        value (float, int, fem.Constant or callable): the value of the soure
        species (festim.Species): the species to which the source is applied
        value_fenics (fem.Function or fem.Constant): the value of the source in
            fenics format
        source_expr (fem.Expression): the expression of the source term that is
            used to update the value_fenics
        time_dependent (bool): True if the value of the source is time dependent
        temperature_dependent (bool): True if the value of the source is temperature
            dependent

    Usage:
        >>> from festim import Source
        >>> Source(volume=my_vol, value=1, species="H")
        >>> Source(volume=my_vol, value=lambda x: 1 + x[0], species="H")
        >>> Source(volume=my_vol, value=lambda t: 1 + t, species="H")
        >>> Source(volume=my_vol, value=lambda T: 1 + T, species="H")
        >>> Source(volume=my_vol, value=lambda x, t: 1 + x[0] + t, species="H")
    """

    def __init__(self, value, volume):
        self.value = value
        self.volume = volume

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value is None:
            self._value = value
        elif isinstance(value, (float, int, fem.Constant, fem.Function)):
            self._value = F.Value(value)
        elif callable(value):
            self._value = F.Value(value)
        else:
            raise TypeError(
                "Value must be a float, int, fem.Constant, fem.Function, or callable"
            )

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        # check that volume is festim.VolumeSubdomain
        if not isinstance(value, F.VolumeSubdomain):
            raise TypeError("volume must be of type festim.VolumeSubdomain")
        self._volume = value


class ParticleSource(SourceBase):
    def __init__(self, value, volume, species):
        self.species = species
        super().__init__(value, volume)

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        # check that species is festim.Species or list of festim.Species
        if not isinstance(value, F.Species):
            raise TypeError("species must be of type festim.Species")

        self._species = value


class HeatSource(SourceBase):
    def __init__(self, value, volume):
        super().__init__(value, volume)

        if self.value.temperature_dependent:
            raise ValueError("Heat source cannot be temperature dependent")
