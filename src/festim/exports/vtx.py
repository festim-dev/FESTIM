import inspect
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Union

import ufl
from dolfinx import fem, io

from festim.helpers import get_interpolation_points
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
        times: list[float] | list[int] | None | None = None,
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
        times: list[float] | list[int] | None | None = None,
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
        times: list[float] | list[int] | None | None = None,
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


class CustomField(ExportBaseClass):
    """Export a custom field to a VTX file

    Args:
        filename: The name of the output file
        expression: A function evaluating the custom field. Positional
            arguments of the function can be "t" (time), "x" (spatial coordinate),
            "T" (temperature), or any key from the `species_dependent_value` dictionary.
        species_dependent_value: A dictionary mapping argument names
            in `expression` to Species objects. Defaults to None.
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps. Defaults to None.
        subdomain: The volume subdomain on which the custom
            field is evaluated. Defaults to None.
        checkpoint: If True, the export will be a checkpoint file using
            adios4dolfinx and won't be readable by ParaView. Default is False.

    Attributes:
        filename: The name of the output file
        expression: A function evaluating the custom field.
        species_dependent_value: A dictionary mapping argument names to Species objects.
        subdomain: The volume subdomain on which the custom field is evaluated.
        checkpoint: If True, the export will be a checkpoint file.
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps.
        function: the function containing the custom field values
        writer: The VTXWriter object used to write the file
        dolfinx_expression: the dolfinx expression used to evaluate the function
    """

    function: fem.Function
    writer: io.VTXWriter
    dolfinx_expression: fem.Expression
    expression: Callable
    species_dependent_value: dict[str, Species]
    subdomain: VolumeSubdomain
    checkpoint: bool

    def __init__(
        self,
        filename: Union[str, Path],
        expression: Callable,
        species_dependent_value: Union[dict[str, Species], None] = None,
        times: Union[list[float], list[int], None] = None,
        subdomain: VolumeSubdomain = None,
        checkpoint: bool = False,
    ):
        super().__init__(
            filename=filename,
            times=times,
            ext=".bp",
        )
        self.expression = expression
        self.species_dependent_value = species_dependent_value or {}
        self.checkpoint = checkpoint
        self.subdomain = subdomain

    def set_dolfinx_expression(
        self,
        temperature: fem.Constant | fem.Function,
        time: fem.Constant,
    ):
        """
        Set the dolfinx expression used to evaluate the custom field. This is done by
        evaluating the user-provided expression with the appropriate arguments and using
        the result to create a dolfinx expression.

        Args:
            temperature: The temperature field to use in the expression
            time: The time to use in the expression
        """
        # check if we are in a mixed domain/discontinuous case
        mixed_domain = any(
            spe.subdomain_to_post_processing_solution
            for spe in self.species_dependent_value.values()
        ) or (self.subdomain.sub_T if self.subdomain else None)

        # get the arguments of the user-provided expression
        arguments = inspect.signature(self.expression).parameters

        # create a dictionary mapping the arguments to the appropriate values
        kwargs = {}
        if "t" in arguments:
            kwargs["t"] = time
        if "x" in arguments:
            x = ufl.SpatialCoordinate(self.function.function_space.mesh)
            kwargs["x"] = x
        if "T" in arguments:
            if isinstance(temperature, fem.Function) and mixed_domain:
                # fem.Function in mixed domain/discontinuous case, use sub_T
                # NOTE I'm not sure that sub_T is updated at every time step
                kwargs["T"] = self.subdomain.sub_T
            else:
                # else use the provided temperature
                kwargs["T"] = temperature
        # check if there are other arguments and if they are in species_dependent_value
        for arg in arguments:
            if arg in self.species_dependent_value:
                spe = self.species_dependent_value[arg]
                if mixed_domain:
                    kwargs[arg] = spe.subdomain_to_post_processing_solution[
                        self.subdomain
                    ]
                else:
                    kwargs[arg] = spe.post_processing_solution
            assert kwargs[arg] is not None, (
                f"Argument {arg} not found in species_dependent_value"
            )

        self.check_valid_inputs(kwargs, mixed_domain)

        # evaluate the user-provided expression with the appropriate arguments and create a
        # dolfinx.fem.Expression
        self.dolfinx_expression = fem.Expression(
            self.expression(**kwargs),
            get_interpolation_points(self.function.function_space.element),
        )

    def check_valid_inputs(self, kwargs: dict, mixed_domain: bool):
        """
        Check if we are in the mixed domain/discontinuous case and if the user-provided
        expression is valid in this case.
        dolfinx.fem.Expression does not support co-dim 0 submeshes and time is defined
        on the parent mesh, so we cannot have time-dependent custom fields in the mixed
        domain/discontinuous case.

        When https://github.com/FEniCS/dolfinx/issues/3207 is resolved we should be
        able to support this
        """

        # check the domain of all kwargs and check that they are the same

        if mixed_domain and "t" in kwargs:
            raise NotImplementedError(
                "Time-dependent custom fields are not implemented in the case of a "
                "mixed domain/discontinuous case."
                "dolfinx.fem.Expression does not support co-dim 0 submeshes and time is"
                "defined on the parent mesh."
                "See https://github.com/FEniCS/dolfinx/issues/3207 for more details."
            )
