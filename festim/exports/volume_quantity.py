import festim as F
import csv
import os


class VolumeQuantity:
    """Export VolumeQuantity

    Args:
        field (festim.Species): species for which the surface flux is computed
        surface (festim.VolumeSubdomain): volume subdomain
        filename (str, optional): name of the file to which the surface flux is exported

    Attributes:
        field (festim.Species): species for which the surface flux is computed
        volume (festim.VolumeSubdomain): volume subdomain
        filename (str): name of the file to which the surface flux is exported
        t (list): list of time values
        data (list): list of values of the surface quantity
    """

    def __init__(self, field, volume, filename: str = None) -> None:
        self.field = field
        self.volume = volume
        self.filename = filename

        self.t = []
        self.data = []

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if value is None:
            self._filename = None
        elif not isinstance(value, str):
            raise TypeError("filename must be of type str")
        elif not value.endswith(".csv") and not value.endswith(".txt"):
            raise ValueError("filename must end with .csv or .txt")
        self._filename = value

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, value):
        # check that field is festim.Species
        if not isinstance(value, (F.Species, str)):
            raise TypeError("field must be of type festim.Species")

        self._field = value

    def write(self, t):
        """If the filename doesnt exist yet, create it and write the header,
        then append the time and value to the file"""

        if not os.path.isfile(self.filename):
            title = "Total volume {}: {}".format(self.volume.id, self.field.name)

            if self.filename is not None:
                with open(self.filename, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["t(s)", f"{title}"])

        with open(self.filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([t, self.value])
