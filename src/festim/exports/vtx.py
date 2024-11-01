import warnings
from pathlib import Path

from dolfinx.fem import Function as _Function

from festim.species import Species as _Species
from festim.subdomain.volume_subdomain import (
    VolumeSubdomain as _VolumeSubdomain,
)


class ExportBaseClass:
    _filename: Path | str

    def __init__(self, filename: str | Path, ext: str) -> None:
        name = Path(filename)
        if name.suffix != ext:
            warnings.warn(
                f"Filename {filename} does not have {ext} extension, adding it."
            )
            name = name.with_suffix(ext)

        self._filename = name

    @property
    def filename(self):
        return self._filename


class VTXTemperatureExport(ExportBaseClass):
    def __init__(self, filename: str | Path):
        super().__init__(filename, ".bp")


class VTXSpeciesExport(ExportBaseClass):
    """Export functions to VTX file

    Args:
        filename: The name of the output file
        field: Set of species to export
        subdomain: A field can be defined on multiple domains.
            This arguments specifies what subdomains we export on.
            If `None` we export on all domains.

    """

    field: list[_Species]
    _subdomain: _VolumeSubdomain

    def __init__(
        self,
        filename: str | Path,
        field: _Species | list[_Species],
        subdomain: _VolumeSubdomain = None,
    ) -> None:
        super().__init__(filename, ".bp")
        self.field = field
        self._subdomain = subdomain

    @property
    def field(self) -> list[_Species]:
        return self._field

    @field.setter
    def field(self, value: _Species | list[_Species]):
        """
        Update the field to export.

        Note:
            This also creates a new writer with the updated field.

        Args:
            value: The species to export

        Raises:
            TypeError: If input field is not a Species or a list of Species
        """
        # check that all elements of list are festim.Species
        if isinstance(value, list):
            for element in value:
                if not isinstance(element, (_Species, str)):
                    raise TypeError(
                        "field must be of type festim.Species or a list of festim.Species or str"
                    )
            val = value
        elif isinstance(value, _Species):
            val = [value]
        else:
            raise TypeError(
                "field must be of type festim.Species or a list of festim.Species or str",
                f"got {type(value)}.",
            )
        self._field = val

    def get_functions(self) -> list[_Function]:
        """
        Returns list of species for a given subdomain.
        If using legacy mode, return the whole species.
        """

        legacy_output: bool = False
        for field in self._field:
            if field.legacy:
                legacy_output = True
                break
        if legacy_output:
            return [field.post_processing_solution for field in self._field]
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
