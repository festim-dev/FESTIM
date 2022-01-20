import FESTIM
import fenics as f


class Export:
    def __init__(self, field=None) -> None:
        self.field = field
        self.function = None


class Exports:
    def __init__(self, exports=[]) -> None:
        self.exports = exports
        self.t = None
        self.V_DG1 = None
        self.final_time = None
        self.nb_iterations = 0

    def write(self, label_to_function,  dt):
        for export in self.exports:
            if isinstance(export, FESTIM.DerivedQuantities):

                # compute derived quantities
                if self.nb_iterations % export.nb_iterations_between_compute == 0:
                    # check if function has to be projected
                    for quantity in export.derived_quantities:
                        if isinstance(quantity, (FESTIM.MaximumVolume, FESTIM.MinimumVolume)):
                            if not isinstance(label_to_function[quantity.field], f.Function):
                                label_to_function[quantity.field] = f.project(label_to_function[quantity.field], self.V_DG1)
                        quantity.function = label_to_function[quantity.field]
                    export.compute(self.t)
                # export derived quantities
                if FESTIM.is_export_derived_quantities(export, self.t, self.final_time, self.nb_iterations):
                    export.write()

            elif isinstance(export, FESTIM.XDMFExport):
                if FESTIM.is_export_xdmf(export, self.t, self.final_time, self.nb_iterations):
                    if export.field == "retention":
                        # if not a Function, project it onto V_DG1
                        if not isinstance(label_to_function["retention"], f.Function):
                            label_to_function["retention"] = f.project(label_to_function["retention"], self.V_DG1)
                    export.function = label_to_function[export.field]
                    export.write(self.t)
                    export.append = True

            elif isinstance(export, FESTIM.TXTExport):
                # if not a Function, project it onto V_DG1
                if not isinstance(label_to_function[export.field], f.Function):
                    label_to_function[export.field] = f.project(label_to_function[export.field], self.V_DG1)
                export.function = label_to_function[export.field]
                export.write(self.t, dt)
        self.nb_iterations += 1
