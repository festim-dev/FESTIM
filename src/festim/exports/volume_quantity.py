import csv
import os

from festim import species


class VolumeQuantity:
    """Export VolumeQuantity

    Args:
        field (festim.Species): species for which the volume quantity is computed
        volume (festim.VolumeSubdomain): volume subdomain
        filename (str, optional): name of the file to which the volume quantity is exported

    Attributes:
        field (festim.Species): species for which the volume quantity is computed
        volume (festim.VolumeSubdomain): volume subdomain
        filename (str): name of the file to which the volume quantity is exported
        t (list): list of time values
        data (list): list of values of the volume quantity
    """

    def __init__(self, field, volume, filename: str = None) -> None:
        self.field = field
        self.volume = volume
        self.filename = filename

        self.t = []
        self.data = []
        self._first_time_export = True

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
        if not isinstance(value, (species.Species, str)):
            raise TypeError("field must be of type festim.Species")

        self._field = value

    def write(self, t):
        """If the filename doesnt exist yet, create it and write the header,
        then append the time and value to the file"""

        if self.filename is not None:
            if self._first_time_export:
                header = ["t(s)", f"{self.title}"]
                with open(self.filename, mode="w+", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
                self._first_time_export = False
            with open(self.filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([t, self.value])
