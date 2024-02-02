import fenics as f
import numpy as np
import festim
import warnings
import os

warnings.simplefilter("always", DeprecationWarning)


class TXTExport(festim.Export):
    """
    Args:
        field (str): the exported field ("solute", "1", "retention",
            "T"...)
        filename (str): the filename (must end with .txt).
        times (list, optional): if provided, the field will be
            exported at these timesteps. Otherwise exports at all
            timesteps. Defaults to None.
        header_format (str, optional): the format of column headers.
            Defautls to ".2e".
    """

    def __init__(self, field, filename, times=None, header_format=".2e") -> None:
        super().__init__(field=field)
        if times:
            self.times = sorted(times)
        else:
            self.times = times
        self.filename = filename
        self.header_format = header_format
        self._first_time = True

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("filename must be a string")
            if not value.endswith(".txt"):
                raise ValueError("filename must end with .txt")
        self._filename = value

    def is_it_time_to_export(self, current_time):
        if self.times is None:
            return True
        for time in self.times:
            if np.isclose(time, current_time, atol=0):
                return True
        return False

    def when_is_next_time(self, current_time):
        if self.times is None:
            return None
        for time in self.times:
            if current_time < time:
                return time
        return None

    def write(self, current_time, steady):
        # create a DG1 functionspace
        V_DG1 = f.FunctionSpace(self.function.function_space().mesh(), "DG", 1)

        solution = f.project(self.function, V_DG1)
        solution_column = np.transpose(solution.vector()[:])
        if self.is_it_time_to_export(current_time):
            # if the directory doesn't exist
            # create it
            dirname = os.path.dirname(self.filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

            # if steady or it is the first time to export
            # write data
            # else append new column to the existing file
            if steady or self._first_time:
                if steady:
                    header = "x,t=steady"
                else:
                    header = f"x,t={format(current_time, self.header_format)}s"
                x = f.interpolate(f.Expression("x[0]", degree=1), V_DG1)
                x_column = np.transpose([x.vector()[:]])
                data = np.column_stack([x_column, solution_column])
                self._first_time = False
            else:
                # Update the header
                old_file = open(self.filename)
                old_header = old_file.readline().split("\n")[0]
                old_file.close()
                header = old_header + f",t={format(current_time, self.header_format)}s"
                # Append new column
                old_columns = np.loadtxt(self.filename, delimiter=",", skiprows=1)
                data = np.column_stack([old_columns, solution_column])

            np.savetxt(self.filename, data, header=header, delimiter=",", comments="")


class TXTExports:
    """
    Args:
        fields (list): list of exported fields ("solute", "1", "retention",
            "T"...)
        filenames (list): list of the filenames for each field (must end with .txt).
        times (list, optional): if provided, fields will be
            exported at these timesteps. Otherwise exports at all
            timesteps. Defaults to None.
        header_format (str, optional): the format of column headers.
            Defautls to ".2e".
    """

    def __init__(
        self, fields=[], filenames=[], times=None, header_format=".2e"
    ) -> None:
        msg = "TXTExports class will be deprecated in future versions of FESTIM"
        warnings.warn(msg, DeprecationWarning)

        self.fields = fields
        if len(self.fields) != len(filenames):
            raise ValueError(
                "Number of fields to be exported "
                "doesn't match number of filenames in txt exports"
            )
        if times:
            self.times = sorted(times)
        else:
            self.times = times
        self.filenames = filenames
        self.header_format = header_format
        self.exports = []
        for function, filename in zip(self.fields, self.filenames):
            self.exports.append(TXTExport(function, filename, times, header_format))
