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

    def __init__(
        self, field, filename, times=None, write_at_last=False, header_format=".2e"
    ) -> None:
        super().__init__(field=field)
        if times:
            self.times = sorted(times)
        else:
            self.times = times
        self.filename = filename
        self.write_at_last = write_at_last
        self.header_format = header_format
        self._first_time = True

        self.data = None
        self.header = None

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

    def is_last(self, current_time, final_time):
        if final_time is None:
            # write if steady
            return True
        elif self.times is None:
            if np.isclose(current_time, final_time, atol=0):
                # write at final time if exports at each timestep
                return True
        else:
            if np.isclose(current_time, self.times[-1], atol=0):
                # write at final time if exports at specific times
                return True
        return False

    def filter_duplicates(self, data, materials):
        x = data[:, 0]

        # Collect all borders
        borders = []
        for material in materials:
            for border in material.borders:
                borders.append(border)
        borders = np.unique(borders)

        # Find indices of the closest duplicates to interfaces
        border_indx = []
        for border in borders:
            closest_indx = np.abs(x - border).argmin()
            closest_x = x[closest_indx]
            for ind in np.where(x == closest_x)[0]:
                border_indx.append(ind)

        # Find indices of first elements in duplicated pairs and mesh borders
        _, unique_indx = np.unique(x, return_index=True)

        # Combine both arrays of indices
        combined_indx = np.concatenate([border_indx, unique_indx])

        # Sort unique indices to return a slice
        combined_indx = sorted(np.unique(combined_indx))

        return data[combined_indx, :]

    def write(self, current_time, final_time, materials, chemical_pot):
        # create a DG1 functionspace if chemical_pot is True
        # else create a CG1 functionspace
        if chemical_pot:
            V = f.FunctionSpace(self.function.function_space().mesh(), "DG", 1)
        else:
            V = f.FunctionSpace(self.function.function_space().mesh(), "CG", 1)

        solution = f.project(self.function, V)
        solution_column = np.transpose(solution.vector()[:])
        if self.is_it_time_to_export(current_time):
            # if the directory doesn't exist
            # create it
            dirname = os.path.dirname(self.filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

            # create header if steady or it is the first time to export
            # else append new column to the existing file
            if final_time is None or self._first_time:
                if final_time is None:
                    self.header = "x,t=steady"
                else:
                    self.header = f"x,t={format(current_time, self.header_format)}s"

                x = f.interpolate(f.Expression("x[0]", degree=1), V)
                x_column = np.transpose([x.vector()[:]])
                self.data = np.column_stack([x_column, solution_column])
                self._first_time = False
            else:
                # Update the header
                self.header += f",t={format(current_time, self.header_format)}s"
                # Add new column
                self.data = np.column_stack([self.data, solution_column])

            if (
                self.write_at_last and self.is_last(current_time, final_time)
            ) or not self.write_at_last:
                if self.is_last(current_time, final_time):
                    # Sort data by the x-column before the last export time
                    self.data = self.data[self.data[:, 0].argsort()]

                    # Filter duplicates if chemical_pot is True
                    if chemical_pot:
                        self.data = self.filter_duplicates(self.data, materials)

                # Write data
                np.savetxt(
                    self.filename,
                    self.data,
                    header=self.header,
                    delimiter=",",
                    comments="",
                )
