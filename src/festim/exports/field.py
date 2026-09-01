import inspect
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Union

import dolfinx
import ufl
from dolfinx import fem

from festim import k_B as _k_B
from festim.reaction import ArrheniusReaction
from festim.species import ImplicitSpecies, Species
from festim.subdomain.volume_subdomain import VolumeSubdomain

from .writers import (
    CheckpointFieldWriter,
    FieldWriter,
    VTKHDFFieldWriter,
    VTXFieldWriter,
    XDMFFieldWriter,
)

_FORMAT_TO_WRITER: dict[str, type[FieldWriter]] = {
    "vtx": VTXFieldWriter,
    "vtkhdf": VTKHDFFieldWriter,
    "checkpoint": CheckpointFieldWriter,
    "xdmf": XDMFFieldWriter,
}


def _extension_for(format: str, backend: str | None) -> str:
    """Return the file extension a given format is written with."""
    if format == "checkpoint":
        return ".h5" if backend == "h5py" else ".bp"
    return {"vtx": ".bp", "vtkhdf": ".vtkhdf", "xdmf": ".xdmf"}[format]


def _resolve_checkpoint_kwarg(format: str, checkpoint: bool) -> str:
    """Map the deprecated ``checkpoint=True`` argument onto ``format``."""
    if not checkpoint:
        return format
    warnings.warn(
        "`checkpoint=True` is deprecated, use `format='checkpoint'` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return "checkpoint"


class FieldExportBase:
    """Base class for exports of fields to a file.

    Args:
        filename: The name of the output file. If its extension doesn't match the
            chosen format, the correct one is substituted and a warning is issued.
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps. Defaults to None.
        format: the output format. One of:

            - ``"vtx"``: ``.bp``, readable by ParaView (the default)
            - ``"vtkhdf"``: ``.vtkhdf``, readable by ParaView. A single scalable HDF5
              file; several exports sharing a filename become blocks of one file.
            - ``"checkpoint"``: for reloading with
              :func:`festim.read_function_from_file`, not readable by ParaView
            - ``"xdmf"``: ``.xdmf`` + ``.h5``, readable by ParaView

        backend: only used by ``format="checkpoint"``: ``"adios2"`` (default, ``.bp``)
            or ``"h5py"`` (``.h5``).

    Attributes:
        filename: The name of the output file
        times: The timesteps to export at, or None for all timesteps
        format: The output format
        backend: The io4dolfinx backend, for ``format="checkpoint"``
        writer: The :class:`festim.exports.writers.FieldWriter` doing the writing
    """

    _filename: Path
    format: str
    backend: str | None
    writer: FieldWriter | None
    #: subdomain the export lives on, overridden by subclasses that have one
    subdomain: VolumeSubdomain | None = None

    def __init__(
        self,
        filename: str | Path,
        times: list[float] | list[int] | None = None,
        format: str = "vtx",
        backend: str | None = None,
    ):
        if format not in _FORMAT_TO_WRITER:
            raise ValueError(
                f"Unknown format {format!r}, expected one of "
                f"{sorted(_FORMAT_TO_WRITER)}."
            )
        self.format = format
        self.backend = backend
        self.writer = None

        ext = _extension_for(format, backend)
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

    def define_writer(
        self,
        functions: list[fem.Function],
        names: list[str],
        mesh: dolfinx.mesh.Mesh,
        block_name: str = "mesh",
        overwrite: bool = True,
    ) -> None:
        """Create the underlying writer for this export's format.

        Args:
            functions: the functions to write at every timestep
            names: the name to store each function under, one per function
            mesh: the mesh the functions live on
            block_name: name of the block, for formats holding several meshes per file
            overwrite: `False` if another export already initialised this file
        """
        writer_cls = _FORMAT_TO_WRITER[self.format]
        if self.format == "checkpoint":
            self.writer = writer_cls(self.filename, backend=self.backend or "adios2")
        else:
            self.writer = writer_cls(self.filename)
        self.writer.initialise(
            functions, names, mesh, block_name=block_name, overwrite=overwrite
        )

    def update(self) -> None:
        """Refresh the data to write. Called before each write; no-op by default."""

    def write(self, t: float) -> None:
        """Write the fields at time `t`."""
        self.writer.write(t)

    def close(self) -> None:
        """Release the underlying file."""
        if self.writer is not None:
            self.writer.close()


class TemperatureExport(FieldExportBase):
    """Export the temperature field to a file.

    Args:
        filename: The name of the output file
        format: The output format, see :class:`FieldExportBase`. Defaults to ``"vtx"``.
        backend: The io4dolfinx backend, for ``format="checkpoint"``.
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps. Defaults to None.

    Attributes:
        filename: The name of the output file
        times: The timesteps to export at, or None for all timesteps
        writer: The writer object used to write the file
    """

    def __init__(
        self,
        filename: str | Path,
        format: str = "vtx",
        backend: str | None = None,
        times: list[float] | list[int] | None = None,
    ):
        super().__init__(filename, times=times, format=format, backend=backend)


class SpeciesExport(FieldExportBase):
    """Export species concentration fields to a file.

    Args:
        filename: The name of the output file
        field: Set of species to export
        subdomain: A field can be defined on multiple domains. This arguments specifies
            what subdomains we export on. If `None` we export on all domains.
        format: The output format, see :class:`FieldExportBase`. Defaults to ``"vtx"``.
        backend: The io4dolfinx backend, for ``format="checkpoint"``.
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps. Defaults to None.

    Attributes:
        filename: The name of the output file
        field: Set of species to export
        subdomain: The subdomain the species are exported on
        times: The timesteps to export at, or None for all timesteps
        writer: The writer object used to write the file

    Example:

        .. code-block:: python

            # one ParaView-readable .vtkhdf file
            F.SpeciesExport("results.vtkhdf", field=[H], subdomain=vol,
                            format="vtkhdf")

            # a checkpoint, to restart from later
            F.SpeciesExport("state.bp", field=[H], subdomain=vol,
                            format="checkpoint")
    """

    def __init__(
        self,
        filename: str | Path,
        field: Species | list[Species],
        subdomain: VolumeSubdomain = None,
        format: str = "vtx",
        backend: str | None = None,
        times: list[float] | list[int] | None = None,
    ):
        super().__init__(filename, times=times, format=format, backend=backend)
        self.field = field
        self.subdomain = subdomain

    @property
    def _checkpoint(self) -> bool:
        """Kept so the legacy problem classes keep working unchanged."""
        return self.format == "checkpoint"

    @property
    def field(self) -> list[Species]:
        return self._field

    @field.setter
    def field(self, value: Species | list[Species]):
        """Update the field to export.

        Args:
            value: The species to export

        Raises:
            TypeError: If input field is not a Species or a list of Species
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
        """Returns list of species for a given subdomain.

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
            if self.subdomain is None:
                raise ValueError("Subdomain must be specified")
            else:
                outfiles = []
                for field in self._field:
                    if self.subdomain in field.subdomains:
                        outfiles.append(
                            field.subdomain_to_post_processing_solution[self.subdomain]
                        )
                return outfiles


class CustomFieldExport(FieldExportBase):
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
        format: The output format, see :class:`FieldExportBase`. Defaults to ``"vtx"``.
        backend: The io4dolfinx backend, for ``format="checkpoint"``.
        checkpoint: Deprecated, use ``format="checkpoint"``.

    Attributes:
        filename: The name of the output file
        expression: A function evaluating the custom field.
        species_dependent_value: A dictionary mapping argument names to Species objects.
        subdomain: The volume subdomain on which the custom field is evaluated.
        checkpoint: True if the export is a checkpoint file.
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps.
        function: the function containing the custom field values
        writer: The writer object used to write the file
        dolfinx_expression: the dolfinx expression used to evaluate the function
    """

    function: fem.Function
    dolfinx_expression: fem.Expression
    expression: Callable
    species_dependent_value: dict[str, Species]
    subdomain: VolumeSubdomain

    def __init__(
        self,
        filename: Union[str, Path],
        expression: Callable,
        species_dependent_value: Union[dict[str, Species], None] = None,
        times: Union[list[float], list[int], None] = None,
        subdomain: VolumeSubdomain = None,
        format: str = "vtx",
        backend: str | None = None,
        checkpoint: bool = False,
    ):
        format = _resolve_checkpoint_kwarg(format, checkpoint)
        super().__init__(
            filename=filename,
            times=times,
            format=format,
            backend=backend,
        )
        self.expression = expression
        self.species_dependent_value = species_dependent_value or {}
        self.subdomain = subdomain

    @property
    def checkpoint(self) -> bool:
        return self.format == "checkpoint"

    def update(self):
        """Re-evaluate the custom field before writing."""
        self.function.interpolate(self.dolfinx_expression)

    @property
    def mixed_domain(self) -> bool:
        """
        Check if we are in a mixed domain/discontinuous case. This is the case if at
        least one of the species in species_dependent_value is defined on a subdomain
        or if the custom field is defined on a subdomain.

        Returns:
            True if we are in a mixed domain/discontinuous case, False otherwise.
        """
        all_explicit_species = [
            spe
            for spe in self.species_dependent_value.values()
            if isinstance(spe, Species)
        ]
        return any(
            spe.subdomain_to_post_processing_solution for spe in all_explicit_species
        ) or (self.subdomain.sub_T if self.subdomain else None)

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
            if isinstance(temperature, fem.Function) and self.mixed_domain:
                # fem.Function in mixed domain/discontinuous case, use sub_T
                # NOTE I'm not sure that sub_T is updated at every time step
                kwargs["T"] = self.subdomain.sub_T
            else:
                # else use the provided temperature
                kwargs["T"] = temperature

        # check if there are other arguments and if they are in species_dependent_value
        for arg in arguments:
            if arg in self.species_dependent_value:
                kwargs[arg] = self._get_species_function(
                    self.species_dependent_value[arg]
                )
            assert kwargs[arg] is not None, (
                f"Argument {arg} not found in species_dependent_value"
            )

        self.check_valid_inputs(kwargs)

        # evaluate the user-provided expression with the appropriate arguments and
        # create a dolfinx.fem.Expression
        self.dolfinx_expression = fem.Expression(
            self.expression(**kwargs),
            self.function.function_space.element.interpolation_points,
        )

    def _get_species_function(self, spe: Species):
        if isinstance(spe, ImplicitSpecies):
            if self.mixed_domain:
                return spe.concentration_submesh(self.subdomain)
            else:
                return spe.concentration
        else:
            if self.mixed_domain:
                return spe.subdomain_to_post_processing_solution[self.subdomain]
            else:
                return spe.post_processing_solution

    def check_valid_inputs(self, kwargs: dict):
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

        if self.mixed_domain and "t" in kwargs:
            raise NotImplementedError(
                "Time-dependent custom fields are not implemented in the case of a "
                "mixed domain/discontinuous case."
                "dolfinx.fem.Expression does not support co-dim 0 submeshes and time is"
                "defined on the parent mesh."
                "See https://github.com/FEniCS/dolfinx/issues/3207 for more details."
            )


class ReactionRateExport(CustomFieldExport):
    """Export a reaction rate to a VTX file

    Args:
        reaction: The reaction to export the rate of.
        filename: The name of the output file.
        direction: The direction of the reaction to export.
            Can be "forward", "backward" or "both". Defaults to "both".
        times: if provided, the field will be exported at these timesteps. Otherwise
            exports at all timesteps. Defaults to None.
        subdomain: The volume subdomain on which the reaction
            rate is evaluated. Defaults to None.
        format: The output format, see :class:`FieldExportBase`. Defaults to ``"vtx"``.
        backend: The io4dolfinx backend, for ``format="checkpoint"``.
        checkpoint: Deprecated, use ``format="checkpoint"``.
    """

    def __init__(
        self,
        reaction: ArrheniusReaction,
        filename: str | Path,
        direction: str = "both",
        times: list[float] | None = None,
        subdomain: VolumeSubdomain | None = None,
        format: str = "vtx",
        backend: str | None = None,
        checkpoint: bool = False,
    ):

        reactant_names = [reactant.name for reactant in reaction.reactant]
        if isinstance(reaction.product, list):
            product_names = [product.name for product in reaction.product]
        else:
            product_names = [reaction.product.name]

        def expression(T, **kwargs):
            _reactant_names = [kwargs[name] for name in reactant_names]
            _product_names = [kwargs[name] for name in product_names]
            k = reaction.k_0 * ufl.exp(-reaction.E_k / (_k_B * T))
            if reaction.p_0 and reaction.E_p:
                p = reaction.p_0 * ufl.exp(-reaction.E_p / (_k_B * T))
            elif reaction.p_0:
                p = reaction.p_0
            else:
                p = 0.0

            forward = k * ufl.product(_reactant_names)
            backward = p * ufl.product(_product_names)

            if direction == "forward":
                return forward
            elif direction == "backward":
                return backward
            else:
                return forward - backward

        self.override_signature(expression, reactant_names, product_names)

        reaction_products = (
            reaction.product
            if isinstance(reaction.product, list)
            else [reaction.product]
        )

        super().__init__(
            filename=filename,
            expression=expression,
            species_dependent_value={
                spe.name: spe for spe in reaction.reactant + reaction_products
            },
            times=times,
            subdomain=subdomain,
            format=format,
            backend=backend,
            checkpoint=checkpoint,
        )

    def override_signature(
        self, expression: Callable, reactant_names: list[str], product_names: list[str]
    ):
        """
        Override the signature of the expression function. This is needed to ensure that
        the expression has the correct arguments for set_dolfinx_expression().

        Args:
            expression: The user-provided expression for the reaction rate. The
                arguments of the expression must be T (temperature) and the names of
                the reactants and products.
        """
        sig_params = [inspect.Parameter("T", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        # Use dict.fromkeys to preserve order and remove duplicates
        for name in dict.fromkeys(reactant_names + product_names):
            sig_params.append(
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )
        expression.__signature__ = inspect.Signature(sig_params)

        assert inspect.signature(expression).parameters.keys() == {
            "T",
            *reactant_names,
            *product_names,
        }, (
            "The expression for the reaction rate is automatically generated based on "
            "the reaction provided. The arguments of the expression must be T "
            "(temperature) and the names of the reactants and products. The current "
            "expression has arguments "
            f"{inspect.signature(expression).parameters.keys()} but should have "
            f"arguments T and {reactant_names + product_names}."
        )


class VTXTemperatureExport(TemperatureExport):
    """Export the temperature field to a VTX (``.bp``) file.

    .. deprecated::
        Use :class:`TemperatureExport`, which supports other formats, instead. This
        class will be removed in a future release.
    """

    def __init__(
        self,
        filename: str | Path,
        times: list[float] | list[int] | None = None,
    ):
        warnings.warn(
            "VTXTemperatureExport is deprecated and will be removed in a future "
            "release, use TemperatureExport(..., format='vtx') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(filename, format="vtx", times=times)


class VTXSpeciesExport(SpeciesExport):
    """Export species fields to a VTX (``.bp``) file.

    .. deprecated::
        Use :class:`SpeciesExport`, which supports other formats, instead. This class
        will be removed in a future release.

    Args:
        filename: The name of the output file
        field: Set of species to export
        subdomain: The subdomain to export on
        checkpoint: If True, write a checkpoint instead of a VTX file. Equivalent to
            ``SpeciesExport(..., format="checkpoint")``.
        times: The timesteps to export at, or None for all timesteps
    """

    def __init__(
        self,
        filename: str | Path,
        field: Species | list[Species],
        subdomain: VolumeSubdomain = None,
        checkpoint: bool = False,
        times: list[float] | list[int] | None = None,
    ):
        warnings.warn(
            "VTXSpeciesExport is deprecated and will be removed in a future release, "
            "use SpeciesExport(..., format='vtx') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            filename,
            field,
            subdomain=subdomain,
            format="checkpoint" if checkpoint else "vtx",
            times=times,
        )
