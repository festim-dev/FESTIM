from FESTIM import (
    SurfaceFlux,
    AverageVolume,
    MinimumVolume,
    MaximumVolume,
    TotalVolume,
    TotalSurface,
    DerivedQuantity,
)
import fenics as f
import os
import csv
from typing import Union


class DerivedQuantities:
    def __init__(
        self,
        file=None,
        folder=None,
        nb_iterations_between_compute=1,
        nb_iterations_between_exports=None,
        **derived_quantities
    ) -> None:
        self.file = file
        self.folder = folder
        self.nb_iterations_between_compute = nb_iterations_between_compute
        self.nb_iterations_between_exports = nb_iterations_between_exports
        self.derived_quantities = []
        # TODO remove this
        self.make_derived_quantities(derived_quantities)
        self.data = [self.make_header()]

    def make_derived_quantities(self, derived_quantities):
        for derived_quantity, list_of_prms_dicts in derived_quantities.items():
            if derived_quantity == "surface_flux":
                quantity_class = SurfaceFlux
            elif derived_quantity == "average_volume":
                quantity_class = AverageVolume
            elif derived_quantity == "minimum_volume":
                quantity_class = MinimumVolume
            elif derived_quantity == "maximum_volume":
                quantity_class = MaximumVolume
            elif derived_quantity == "total_volume":
                quantity_class = TotalVolume
            elif derived_quantity == "total_surface":
                quantity_class = TotalSurface
            for prms_dict in list_of_prms_dicts:
                if "volumes" in prms_dict:
                    for entity in prms_dict["volumes"]:
                        self.derived_quantities.append(
                            quantity_class(field=prms_dict["field"], volume=entity)
                        )
                if "surfaces" in prms_dict:
                    for entity in prms_dict["surfaces"]:
                        self.derived_quantities.append(
                            quantity_class(field=prms_dict["field"], surface=entity)
                        )

    def make_header(self):
        header = ["t(s)"]
        for quantity in self.derived_quantities:
            header.append(quantity.title)
        return header

    def assign_measures_to_quantities(self, dx, ds):
        self.volume_markers = dx.subdomain_data()
        for quantity in self.derived_quantities:
            quantity.dx = dx
            quantity.ds = ds
            quantity.n = f.FacetNormal(dx.subdomain_data().mesh())

    def assign_properties_to_quantities(self, materials):
        """Assign properties attributes to all DerivedQuantity objects
        (D, S, thermal_cond and H) based on the properties stored in materials

        Args:
            materials (FESTIM.Materials): the materials
        """
        for quantity in self.derived_quantities:
            quantity.D = materials.D
            quantity.S = materials.S
            quantity.thermal_cond = materials.thermal_cond
            quantity.H = materials.H

    def compute(self, t):

        # TODO need to support for soret flag in surface flux
        row = [t]
        for quantity in self.derived_quantities:
            if isinstance(quantity, (MaximumVolume, MinimumVolume)):
                row.append(quantity.compute(self.volume_markers))
            else:
                row.append(quantity.compute())
        self.data.append(row)

    def write(self):
        if self.file is not None:
            file_export = ""
            if self.folder is not None:
                file_export += self.folder + "/"
                os.makedirs(os.path.dirname(file_export), exist_ok=True)
            if self.file.endswith(".csv"):
                file_export += self.file
            else:
                file_export += self.file + ".csv"
            busy = True
            while busy:
                try:
                    with open(file_export, "w+") as f:
                        busy = False
                        writer = csv.writer(f, lineterminator="\n")
                        for val in self.data:
                            writer.writerows([val])
                except OSError as err:
                    print("OS error: {0}".format(err))
                    print(
                        "The file " + file_export + ".txt might currently be busy."
                        "Please close the application then press any key."
                    )
                    input()
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
                return t >= final_time
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
        """_summary_

        Args:
            surfaces (Union[list, int], optional): _description_. Defaults to None.
            volumes (Union[list, int], optional): _description_. Defaults to None.
            fields (Union[list, str], optional): _description_. Defaults to None.
            instances (DerivedQuantity, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if surfaces is not None and not isinstance(surfaces, list):
            surfaces = [surfaces]
        if volumes is not None and not isinstance(volumes, list):
            volumes = [volumes]
        if fields is not None and not isinstance(fields, list):
            fields = [fields]
        if instances is not None and not isinstance(instances, list):
            instances = [instances]

        quantities = []

        for quantity in self.derived_quantities:
            match_surface, match_volume, match_field, match_instance = (
                False,
                False,
                False,
                False,
            )

            if surfaces is not None:
                if hasattr(quantity, "surface") and quantity.surface in surfaces:
                    match_surface = True
            else:
                match_surface = True

            if volumes is not None:
                if hasattr(quantity, "volume") and quantity.volume in volumes:
                    match_volume = True
            else:
                match_volume = True

            if fields is not None:
                if quantity.field in fields:
                    match_field = True
            else:
                match_field = True

            if instances is not None:
                if isinstance(quantity, tuple(instances)):
                    match_instance = True
            else:
                match_instance = True

            if match_surface and match_volume and match_field and match_instance:
                quantities.append(quantity)

        if len(quantities) == 1:
            quantities = quantities[0]
        return quantities
