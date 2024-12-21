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
        filter (bool): if True and the field is projected to a DG function space,
            the duplicated vertices in the output file array are filtered except those near interfaces.
            Defaults to True.
        write_at_last (bool): if True, the data will be exported at
            the last export time. Otherwise, the data will be exported
            at each export time. Defaults to False.
        header_format (str, optional): the format of column headers.
            Defautls to ".2e".

    Attributes:
        data (np.array): the data array of the exported field. The first column
            is the mesh vertices. Each next column is the field profile at the specific
            export time.
        header (str): the header of the exported file.
        V (fenics.FunctionSpace): the vector-function space for the exported field.

    .. note::
        The exported field is projected to DG if conservation of chemical potential is considered or
        ``traps_element_type`` is "DG".

    """

    def __init__(
        self,
        field,
        filename,
        times=None,
        filter=True,
        write_at_last=False,
        header_format=".2e",
    ) -> None:
        super().__init__(field=field)
        if times:
            self.times = sorted(times)
        else:
            self.times = times
        self.filename = filename
        self.filter = filter
        self.write_at_last = write_at_last
        self.header_format = header_format

        self.data = None
        self.header = None
        self.V = None
        self._unique_indices = None

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
        """
        Checks if the exported field should be written to a file or not
        based on the current time and the ``TXTExport.times``

        Args:
            current_time (float): the current simulation time

        Returns:
            bool: True if the exported field should be written to a file, else False
        """

        if self.times is None:
            return True
        for time in self.times:
            if np.isclose(time, current_time, atol=0):
                return True
        return False

    def is_last(self, current_time, final_time):
        """
        Checks if the current simulation step equals to the last export time.
        based on the final simulation time, ``TXTExport.times``, and the current time

        Args:
            current_time (float): the current simulation time.
            final_time (float, None): the final simulation time.

        Returns:
            bool: True if simulation is steady (final_time is None), if ``TXTExport.times`` are not
            provided and the current time equals to the final time, or if
            ``TXTExport.times`` are provided and the current time equals to the last time in
            ``TXTExport.times``, else False.
        """

        if final_time is None:
            # write if steady
            return True
        elif self.times is None:
            if np.isclose(current_time, final_time, atol=0):
                # write at the final time if exports at each timestep
                return True
        elif np.isclose(current_time, self.times[-1], atol=0):
            # write at the final time if exports at specific times
            return True
        return False

    def initialise(self, mesh, project_to_DG=False, materials=None):
        """
        Initialises ``TXTExport``. Depending on the ``project_to_DG flag``, defines a function space (DG1 or CG1)
        for projection of the exported field. After that, an unsorted array of mesh vertices is created for export.
        The array is then used to obtain indices of sorted elements for the data export.

        .. note::
            If DG1 is used and the ``filter`` flag is True, the duplicated vertices in the array are filtered except those near interfaces.
            The interfaces are defined by ``material.borders`` in the ``Materials`` list.

        Args:
            mesh (fenics.Mesh): the mesh.
            project_to_DG (bool): if True, the exported field is projected to a DG1 function space.
                Defaults to False.
            materials (festim.Materials): the materials. Defaults to None.
        """

        if project_to_DG:
            self.V = f.FunctionSpace(mesh, "DG", 1)
        else:
            self.V = f.FunctionSpace(mesh, "CG", 1)

        x = f.interpolate(f.Expression("x[0]", degree=1), self.V)
        x_column = np.transpose([x.vector()[:]])

        # if filter is True, get indices of duplicates near interfaces
        # and indices of the first elements from a pair of duplicates otherwise
        if project_to_DG and self.filter:
            # Collect all borders
            borders = []
            for material in materials:
                if material.borders:
                    for border in material.borders:
                        borders.append(border)
            borders = np.unique(borders)

            # Find indices of the closest duplicates to interfaces
            border_indices = []
            for border in borders:
                closest_indx = np.abs(x_column - border).argmin()
                closest_x = x_column[closest_indx]
                for ind in np.where(x_column == closest_x)[0]:
                    border_indices.append(ind)

            # Find indices of first elements in duplicated pairs and mesh borders
            _, mesh_indices = np.unique(x_column, return_index=True)

            # Get unique indices from both arrays preserving the order in unsorted x-array
            unique_indices = []
            for indx in np.argsort(x_column, axis=0)[:, 0]:
                if (indx in mesh_indices) or (indx in border_indices):
                    unique_indices.append(indx)

            self._unique_indices = np.array(unique_indices)

        else:
            # Get list of sorted indices
            self._unique_indices = np.argsort(x_column, axis=0)[:, 0]

        self.data = x_column[self._unique_indices]
        self.header = "x"

    def write(self, current_time, final_time):
        """
        Modifies the header and writes the data to a file depending on
        the current and the final times of a simulation.

        Args:
            current_time (float): the current simulation time.
            final_time (float, None): the final simulation time.
        """

        if self.is_it_time_to_export(current_time):
            solution = f.project(self.function, self.V)
            solution_column = np.transpose(solution.vector()[:])

            # if the directory doesn't exist
            # create it
            dirname = os.path.dirname(self.filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

            # if steady, add the corresponding label
            # else append new export time to the header
            steady = final_time is None
            if steady:
                self.header += ",t=steady"
            else:
                self.header += f",t={format(current_time, self.header_format)}s"

            # Add new column of filtered and sorted data
            self.data = np.column_stack(
                [self.data, solution_column[self._unique_indices]]
            )

            if (
                self.write_at_last and self.is_last(current_time, final_time)
            ) or not self.write_at_last:

                # Write data
                np.savetxt(
                    self.filename,
                    self.data,
                    header=self.header,
                    delimiter=",",
                    comments="",
                )
