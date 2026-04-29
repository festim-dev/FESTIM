import inspect
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Union

import ufl
from dolfinx import fem, io

from festim import k_B as _k_B
from festim.helpers import get_interpolation_points
from festim.reaction import Reaction
from festim.species import ImplicitSpecies, Species
from festim.subdomain.volume_subdomain import VolumeSubdomain


class ExportBaseClass:
    """Export functions to VTX file.

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
    """Export temperature field functions to VTX file.

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
    """Export species field functions to VTX file.

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
        """Update the field to export.

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


class CustomFieldExport(ExportBaseClass):
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
            get_interpolation_points(self.function.function_space.element),
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


class ReactionRate(CustomFieldExport):
    def __init__(
        self,
        reaction: Reaction,
        filename: str | Path,
        direction: str = "both",
        times: list[float] | None = None,
        subdomain: VolumeSubdomain | None = None,
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
