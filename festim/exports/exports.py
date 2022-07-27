import festim
import fenics as f


class Exports:
    def __init__(self, exports=[]) -> None:
        self.exports = exports
        self.t = None
        self.V_DG1 = None
        self.final_time = None
        self.nb_iterations = 0

    def write(self, label_to_function, dt):
        for export in self.exports:
            if isinstance(export, festim.DerivedQuantities):

                # compute derived quantities
                if export.is_compute(self.nb_iterations):
                    # check if function has to be projected
                    for quantity in export.derived_quantities:
                        if isinstance(
                            quantity, (festim.MaximumVolume, festim.MinimumVolume)
                        ):
                            if not isinstance(
                                label_to_function[quantity.field], f.Function
                            ):
                                label_to_function[quantity.field] = f.project(
                                    label_to_function[quantity.field], self.V_DG1
                                )
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
                    export.write(self.t)
                    export.append = True

            elif isinstance(export, festim.TXTExport):
                # if not a Function, project it onto V_DG1
                if not isinstance(label_to_function[export.field], f.Function):
                    label_to_function[export.field] = f.project(
                        label_to_function[export.field], self.V_DG1
                    )
                export.function = label_to_function[export.field]
                export.write(self.t, dt)
        self.nb_iterations += 1

    def initialise_derived_quantities(self, dx, ds, materials):
        """If derived quantities in exports, creates header and adds measures
        and properties

        Args:
            dx (fenics.Measure): the measure for dx
            ds (fenics.Measure): the measure for ds
            materials (festim.Materials): the materials
        """
        for export in self.exports:
            if isinstance(export, festim.DerivedQuantities):
                export.data = [export.make_header()]
                export.assign_measures_to_quantities(dx, ds)
                export.assign_properties_to_quantities(materials)
