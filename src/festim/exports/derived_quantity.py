import csv
from abc import ABC, abstractmethod


class DerivedQuantity(ABC):
    """Base class for all derived quantities.

    Attributes:
        filename: name of the file to which the quantity is exported
        t: list of time values
        data: list of values of the quantity
    """

    filename: str | None
    t: list[float]
    data: list[float]

    def __init__(self, filename: str | None = None) -> None:
        self.filename = filename
        self.t = []
        self.data = []
        self._first_time_export = True

    @property
    @abstractmethod
    def title(self):
        pass

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

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

    def write(self, t):
        """If the filename doesnt exist yet, create it and write the header, then append
        the time and value to the file."""

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
