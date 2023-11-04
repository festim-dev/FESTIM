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
        label (str): label of the field. Will also be the filename.
        folder (str): the export folder
        times (list, optional): if provided, the field will be
            exported at these timesteps. Otherwise exports at all
            timesteps. Defaults to None.
    """

    def __init__(self, field, label, folder, times=None) -> None:
        super().__init__(field=field)
        if times:
            self.times = sorted(times)
        else:
            self.times = times
        self.label = label
        self.folder = folder
        self._first_time = True

    @property
    def filename(self):
        return f"{self.folder}/{self.label}.txt"

    def is_it_time_to_export(self, current_time):
        if self.times is None:
            return True
        for time in self.times:
            if np.isclose(time, current_time):
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
            if steady:
                header = "x,t=steady"
            else:
                header = "x,t={}s".format(current_time)

            # if the directory doesn't exist
            # create it
            dirname = os.path.dirname(self.filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

            # if steady or it is the first time to export
            # write data
            # else append new column to the existing file
            if steady or self._first_time:
                x = f.interpolate(f.Expression("x[0]", degree=1), V_DG1)
                x_column = np.transpose([x.vector()[:]])
                data = np.column_stack([x_column, solution_column])
                self._first_time = False
            else:
                # Update the header
                old_file = open(self.filename)
                old_header = old_file.readline().split("\n")[0]
                old_file.close()
                header = old_header + ",t={}s".format(current_time)
                # Append new column
                old_columns = np.loadtxt(self.filename, delimiter=",", skiprows=1)
                data = np.column_stack([old_columns, solution_column])

            np.savetxt(self.filename, data, header=header, delimiter=",", comments="")


class TXTExports:
    def __init__(self, fields=[], times=[], labels=[], folder=None) -> None:
        self.fields = fields
        if len(self.fields) != len(labels):
            raise ValueError(
                "Number of fields to be exported "
                "doesn't match number of labels in txt exports"
            )
        self.times = sorted(times)
        self.labels = labels
        self.folder = folder
        self.exports = []
        for function, label in zip(self.fields, self.labels):
            self.exports.append(TXTExport(function, label, folder, times))
