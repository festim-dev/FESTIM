import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from dolfinx import fem, io

from festim.species import Species
from festim.subdomain.volume_subdomain import VolumeSubdomain


class ExportBaseClass:
    """Export functions to VTX file

    Args:
        filename: The name of the output file
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps. Defaults to None.

    Attributes:
        filename: The name of the output file
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps. Defaults to None.
    """

    _filename: Path | str
    writer: io.VTXWriter

    def __init__(
        self,
        filename: str | Path,
        ext: str,
        times: Optional[list[float] | list[int] | None] = None,
    ):
        name = Path(filename)
        if name.suffix != ext:
            warnings.warn(
                f"Filename {filename} does not have {ext} extension, adding it."
            )
            name = name.with_suffix(ext)

        self._filename = name
        if times:
            self.times = sorted(times)
        else:
            self.times = times

    @property
    def filename(self):
        return self._filename

    def is_it_time_to_export(self, current_time: float) -> bool:
        """
        Checks if the exported field should be written to a file or not based on the
        current time and the times in `export.times`

        Args:
            current_time: the current simulation time

        Returns:
            bool: True if the exported field should be written to a file, else False
        """

        if self.times is None:
            return True

        for time in self.times:
            if np.isclose(time, current_time, atol=0):
                return True

        return False


class VTXTemperatureExport(ExportBaseClass):
    """Export temperature field functions to VTX file

    Args:
        filename: The name of the output file
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps. Defaults to None.

    Attributes:
        filename: The name of the output file
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps. Defaults to None.
        writer: The VTXWriter object used to write the file
    """

    writer: io.VTXWriter

    def __init__(
        self,
        filename: str | Path,
        times: Optional[list[float] | list[int] | None] = None,
    ):
        super().__init__(filename, ".bp", times)


class VTXSpeciesExport(ExportBaseClass):
    """Export species field functions to VTX file

    Args:
        filename: The name of the output file
        field: Set of species to export
        subdomain: A field can be defined on multiple domains. This arguments specifies
            what subdomains we export on. If `None` we export on all domains.
        checkpoint: If True, the export will be a checkpoint file using adios4dolfinx
            and won't be readable by ParaView. Default is False.
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps. Defaults to None.

    Attributes:
        filename: The name of the output file
        field: Set of species to export
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps. Defaults to None.
        writer: The VTXWriter object used to write the file
    """

    _subdomain: VolumeSubdomain
    _checkpoint: bool
    writer: io.VTXWriter

    def __init__(
        self,
        filename: str | Path,
        field: Species | list[Species],
        subdomain: VolumeSubdomain = None,
        checkpoint: bool = False,
        times: Optional[list[float] | list[int] | None] = None,
    ):
        super().__init__(filename, ".bp", times)
        self.field = field
        self._subdomain = subdomain
        self._checkpoint = checkpoint

    @property
    def field(self) -> list[Species]:
        return self._field

    @field.setter
    def field(self, value: Species | list[Species]):
        """
        Update the field to export.

        Args:
            value: The species to export

        Raises:
            TypeError: If input field is not a Species or a list of Species

        Note:
            This also creates a new writer with the updated field.
        """
        # check that all elements of list are festim.Species
        if isinstance(value, list):
            for element in value:
                if not isinstance(element, Species | str):
                    raise TypeError(
                        "field must be of type festim.Species or a list of "
                        "festim.Species or str"
                    )
            val = value
        elif isinstance(value, Species):
            val = [value]
        else:
            raise TypeError(
                "field must be of type festim.Species or a list of festim.Species or "
                "str",
                f"got {type(value)}.",
            )
        self._field = val

    def get_functions(self) -> list[fem.Function]:
        """
        Returns list of species for a given subdomain. If using legacy mode, return the
        whole species.
        """

        legacy_output: bool = False
        for field in self._field:
            if field.legacy:
                legacy_output = True
                break
        if legacy_output:
            return [field.sub_function for field in self._field]
        else:
            if self._subdomain is None:
                raise ValueError("Subdomain must be specified")
            else:
                outfiles = []
                for field in self._field:
                    if self._subdomain in field.subdomains:
                        outfiles.append(
                            field.subdomain_to_post_processing_solution[self._subdomain]
                        )
                return outfiles
