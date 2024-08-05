import festim
import fenics as f
import warnings


class Exports(list):
    """
    A list of festim.Export objects
    """

    def __init__(self, *args):
        # checks that input is list
        if len(args) == 0:
            super().__init__()
        else:
            if not isinstance(*args, list):
                raise TypeError("festim.Exports must be a list")
            super().__init__(self._validate_export(item) for item in args[0])

        self.t = None
        self.V_DG1 = None
        self.final_time = None
        self.nb_iterations = 0

    @property
    def exports(self):
        warnings.warn(
            "The exports attribute will be deprecated in a future release, please use festim.Exports as a list instead",
            DeprecationWarning,
        )
        return self

    @exports.setter
    def exports(self, value):
        warnings.warn(
            "The exports attribute will be deprecated in a future release, please use festim.Exports as a list instead",
            DeprecationWarning,
        )
        if isinstance(value, list):
            if not all(
                (
                    isinstance(t, festim.Export)
                    or isinstance(t, festim.DerivedQuantities)
                )
                for t in value
            ):
                raise TypeError("exports must be a list of festim.Export")
            super().__init__(value)
        else:
            raise TypeError("exports must be a list")

    def __setitem__(self, index, item):
        super().__setitem__(index, self._validate_export(item))

    def insert(self, index, item):
        super().insert(index, self._validate_export(item))

    def append(self, item):
        super().append(self._validate_export(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(self._validate_export(item) for item in other)

    def _validate_export(self, value):
        if isinstance(value, festim.Export) or isinstance(
            value, festim.DerivedQuantities
        ):
            return value
        raise TypeError("festim.Exports must be a list of festim.Export")

    def write(self, label_to_function, dx):
        """writes to file

        Args:
            label_to_function (dict): dictionary of labels mapped to solutions
            dx (fenics.Measure): the measure for dx
        """
        for export in self:
            if isinstance(export, festim.DerivedQuantities):
                # compute derived quantities
                if export.is_compute(self.nb_iterations):
                    # check if function has to be projected
                    for quantity in export:
                        if isinstance(
                            quantity, (festim.MaximumVolume, festim.MinimumVolume)
                        ):
                            if not isinstance(
                                label_to_function[quantity.field], f.Function
                            ):
                                label_to_function[quantity.field] = f.project(
                                    label_to_function[quantity.field], self.V_DG1
                                )
                        if isinstance(quantity, festim.AdsorbedHydrogen):
                            for surf_funcs in label_to_function[quantity.field]:
                                if quantity.surface in surf_funcs["surfaces"]:
                                    ind = surf_funcs["surfaces"].index(quantity.surface)
                                    quantity.function = surf_funcs[
                                        "post_processing_solutions"
                                    ][ind]
                        else:
                            quantity.function = label_to_function[quantity.field]
                    export.compute(self.t)
                # export derived quantities
                if export.is_export(self.t, self.final_time, self.nb_iterations):
                    export.write()

            elif isinstance(export, festim.XDMFExport):
                if export.is_export(self.t, self.final_time, self.nb_iterations):
                    if export.field == "retention":
                        # if not a Function, project it onto V_DG1
                        if not isinstance(label_to_function["retention"], f.Function):
                            label_to_function["retention"] = f.project(
                                label_to_function["retention"], self.V_DG1
                            )
                    export.function = label_to_function[export.field]
                    if isinstance(export, festim.TrapDensityXDMF):
                        export.write(self.t, dx)
                    else:
                        export.write(self.t)
                    export.append = True

            elif isinstance(export, festim.TXTExport):
                # if not a Function, project it onto V_DG1
                if not isinstance(label_to_function[export.field], f.Function):
                    label_to_function[export.field] = f.project(
                        label_to_function[export.field], self.V_DG1
                    )
                export.function = label_to_function[export.field]
                steady = self.final_time == None
                export.write(self.t, steady)
        self.nb_iterations += 1

    def initialise_derived_quantities(self, dx, ds, materials):
        """If derived quantities in exports, creates header and adds measures
        and properties

        Args:
            dx (fenics.Measure): the measure for dx
            ds (fenics.Measure): the measure for ds
            materials (festim.Materials): the materials
        """
        for export in self:
            if isinstance(export, festim.DerivedQuantities):
                # reset the data of the derived quantities
                export.data = []
                export.t = []
                for quantity in export:
                    quantity.t = []
                    quantity.data = []
                export.assign_measures_to_quantities(dx, ds)
                export.assign_properties_to_quantities(materials)
