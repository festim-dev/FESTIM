from festim import (
    MinimumVolume,
    MaximumVolume,
    DerivedQuantity,
)
import fenics as f
import os
import numpy as np
from typing import Union
import warnings


class DerivedQuantities(list):
    """
    A list of festim.DerivedQuantity objects

    Args:
        filename (str, optional): the filename (must end with .csv).
            If None, the data will not be exported. Defaults to None.
        nb_iterations_between_compute (int, optional): number of
            iterations between each derived quantities computation.
            Defaults to 1.
        nb_iterations_between_exports (int, optional): number of
            iterations between each export. If None, the file will be
            exported at the last timestep. Defaults to None.
        show_units (bool, optional): will show the units of each
            derived quantity in the title in export

    Attributes:
        filename (str): the filename.
        nb_iterations_between_compute (int): number of
            iterations between each derived quantities computation.
        nb_iterations_between_exports (int): number of
            iterations between each export. If None, the file will be
            exported at the last timestep.
        show_units (bool): will show the units of each
            derived quantity in the title in export
        data (list): the data to be exported
        t (list): the time steps
    """

    def __init__(
        self,
        *args,
        filename: str = None,
        nb_iterations_between_compute: int = 1,
        nb_iterations_between_exports: int = None,
        show_units=False,
    ) -> None:
        # checks that input is list
        if len(args) == 0:
            super().__init__()
        else:
            if not isinstance(*args, list):
                raise TypeError("festim.DerivedQuantities must be a list")
            super().__init__(self._validate_derived_quantity(item) for item in args[0])

        self.filename = filename
        self.nb_iterations_between_compute = nb_iterations_between_compute
        self.nb_iterations_between_exports = nb_iterations_between_exports
        self.show_units = show_units

        self.data = []
        self.t = []

    @property
    def derived_quantities(self):
        warnings.warn(
            "The derived_quantities attribute will be deprecated in a future release, please use festim.DerivedQuantities as a list instead",
            DeprecationWarning,
        )
        return self

    @derived_quantities.setter
    def derived_quantities(self, value):
        warnings.warn(
            "The derived_quantities attribute will be deprecated in a future release, please use festim.DerivedQuantities as a list instead",
            DeprecationWarning,
        )
        if isinstance(value, list):
            if not all(isinstance(t, DerivedQuantity) for t in value):
                raise TypeError(
                    "derived_quantities must be a list of festim.DerivedQuantity"
                )
            super().__init__(value)
        else:
            raise TypeError("derived_quantities must be a list")

    def __setitem__(self, index, item):
        super().__setitem__(index, self._validate_derived_quantity(item))

    def insert(self, index, item):
        super().insert(index, self._validate_derived_quantity(item))

    def append(self, item):
        super().append(self._validate_derived_quantity(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(self._validate_derived_quantity(item) for item in other)

    def _validate_derived_quantity(self, value):
        if isinstance(value, DerivedQuantity):
            return value
        raise TypeError(
            "festim.DerivedQuantities must be a list of festim.DerivedQuantity"
        )

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("filename must be a string")
            if not value.endswith(".csv"):
                raise ValueError("filename must end with .csv")
        self._filename = value

    def make_header(self):
        header = ["t(s)"]
        for quantity in self:
            quantity.show_units = self.show_units
            if self.show_units is False:
                warnings.warn(
                    "The current derived_quantities title style will be deprecated in a future release, please use show_units=True instead",
                    DeprecationWarning,
                )
            header.append(quantity.title)
        return header

    def assign_measures_to_quantities(self, dx, ds):
        self.volume_markers = dx.subdomain_data()
        for quantity in self:
            quantity.dx = dx
            quantity.ds = ds
            quantity.n = f.FacetNormal(dx.subdomain_data().mesh())

    def assign_properties_to_quantities(self, materials):
        """Assign properties attributes to all DerivedQuantity objects
        (D, S, thermal_cond and H) based on the properties stored in materials

        Args:
            materials (festim.Materials): the materials
        """
        for quantity in self:
            quantity.D = materials.D
            quantity.S = materials.S
            quantity.thermal_cond = materials.thermal_cond
            quantity.Q = materials.Q

    def compute(self, t):
        # TODO need to support for soret flag in surface flux
        row = [t]
        for quantity in self:
            if isinstance(quantity, (MaximumVolume, MinimumVolume)):
                value = quantity.compute(self.volume_markers)
            else:
                value = quantity.compute()

            # check if first time writing data
            if len(self.data) == 0:
                self.data = [self.make_header()]

            quantity.data.append(value)
            quantity.t.append(t)
            row.append(value)
        self.data.append(row)
        self.t.append(t)

    def write(self):
        if self.filename is not None:
            # if the directory doesn't exist
            # create it
            dirname = os.path.dirname(self.filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

            # save the data to csv
            np.savetxt(self.filename, np.array(self.data), fmt="%s", delimiter=",")
        return True

    def is_export(self, t, final_time, nb_iterations):
        """Checks if the derived quantities should be exported or not based on
        the current time, the final time of simulation and the current number
        of iterations

        Args:
            t (float): the current time
            final_time (float): the final time of the simulation
            nb_iterations (int): the current number of time steps

        Returns:
            bool: True if the derived quantities should be exported, else False
        """
        if final_time is not None:
            nb_its_between_exports = self.nb_iterations_between_exports
            if nb_its_between_exports is None:
                # export at the end
                return np.isclose(t, final_time, atol=0)
            else:
                # export every N iterations
                return nb_iterations % nb_its_between_exports == 0
        else:
            # if steady state, export
            return True

    def is_compute(self, nb_iterations):
        """Checks if the derived quantities should be computed or not based on
        the current number of iterations

        Args:
            nb_iterations (int): the current number of time steps

        Returns:
            bool: True if it's time to compute, else False
        """
        return nb_iterations % self.nb_iterations_between_compute == 0

    def filter(
        self,
        surfaces: Union[list, int] = None,
        volumes: Union[list, int] = None,
        fields: Union[list, str] = None,
        instances: DerivedQuantity = None,
    ):
        """Finds DerivedQuantity objects that match surfaces, volumes, and instances.

        Args:
            surfaces (Union[list, int], optional): the surface ids to match.
                Defaults to None.
            volumes (Union[list, int], optional): the volume ids to match.
                Defaults to None.
            fields (Union[list, str], optional): the fields to match.
                Defaults to None.
            instances (DerivedQuantity, optional): the DerivedQuantity
                instances to match. Defaults to None.

        Returns:
            list, DerivedQuantity: if only one quantity matches returns this
                quantity, else returs a list of DerivedQuantity
        """
        # ensure arguments are list
        if surfaces is not None and not isinstance(surfaces, list):
            surfaces = [surfaces]
        if volumes is not None and not isinstance(volumes, list):
            volumes = [volumes]
        if fields is not None and not isinstance(fields, list):
            fields = [fields]
        if instances is not None and not isinstance(instances, list):
            instances = [instances]

        quantities = []

        # iterate through derived_quantities
        for quantity in self:
            # initialise flags to False
            match_surface, match_volume, match_field, match_instance = (
                False,
                False,
                False,
                False,
            )

            # check if matches surface
            if surfaces is not None:
                if hasattr(quantity, "surface") and quantity.surface in surfaces:
                    match_surface = True
            else:
                match_surface = True

            # check if matches volume
            if volumes is not None:
                if hasattr(quantity, "volume") and quantity.volume in volumes:
                    match_volume = True
            else:
                match_volume = True

            # check if matches field
            if fields is not None:
                if quantity.field in fields:
                    match_field = True
            else:
                match_field = True

            # check if matches instance
            if instances is not None:
                if isinstance(quantity, tuple(instances)):
                    match_instance = True
            else:
                match_instance = True

            # if all flags are True, append to the list
            if match_surface and match_volume and match_field and match_instance:
                quantities.append(quantity)

        if len(quantities) == 1:
            quantities = quantities[0]
        return quantities
