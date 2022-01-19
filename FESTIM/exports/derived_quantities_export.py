from FESTIM import Export, R
import fenics as f
import numpy as np
import os
import csv


class DerivedQuantity(Export):
    def __init__(self, field) -> None:
        super().__init__(field=field)
        self.dx = None
        self.ds = None
        self.n = None
        self.D = None
        self.S = None
        self.thermal_cond = None
        self.H = None
        self.T = None


class SurfaceFlux(DerivedQuantity):
    def __init__(self, field, surface) -> None:
        super().__init__(field=field)
        self.surface = surface
        self.title = "Flux surface {}: {}".format(self.surface, self.field)

    def compute(self, soret=False):
        field_to_prop = {
            "0": self.D,
            "solute": self.D,
            0: self.D,
            "T": self.thermal_cond
        }
        self.prop = field_to_prop[self.field]
        flux = f.assemble(self.prop*f.dot(f.grad(self.function), self.n)*self.ds(self.surface))
        if soret and self.field in [0, "0", "solute"]:
            flux += f.assemble(
                        self.prop*self.function*self.H/(R*self.T**2)*f.dot(f.grad(self.T), self.n)*self.ds(self.surface))
        return flux


class HydrogenFlux(SurfaceFlux):
    def __init__(self, surface) -> None:
        super().__init__(field="solute", surface=surface)


class ThermalFlux(SurfaceFlux):
    def __init__(self, surface) -> None:
        super().__init__(field="T", surface=surface)


class AverageVolume(DerivedQuantity):
    def __init__(self, field, volume) -> None:
        super().__init__(field)
        self.volume = volume
        self.title = "Average {} volume {}".format(self.field, self.volume)

    def compute(self):
        return f.assemble(self.function*self.dx(self.volume))/f.assemble(1*self.dx(self.volume))


class MinimumVolume(DerivedQuantity):
    def __init__(self, field, volume) -> None:
        super().__init__(field)
        self.volume = volume
        self.title = "Minimum {} volume {}".format(self.field, self.volume)

    def compute(self, volume_markers):
        '''Minimum of f over subdomains cells marked with self.volume'''
        V = self.function.function_space()

        dm = V.dofmap()

        subd_dofs = np.unique(np.hstack(
            [dm.cell_dofs(c.index())
             for c in f.SubsetIterator(volume_markers, self.volume)]))

        return np.min(self.function.vector().get_local()[subd_dofs])


class MaximumVolume(DerivedQuantity):
    def __init__(self, field, volume) -> None:
        super().__init__(field)
        self.volume = volume
        self.title = "Maximum {} volume {}".format(self.field, self.volume)

    def compute(self, volume_markers):
        '''Minimum of f over subdomains cells marked with self.volume'''
        V = self.function.function_space()

        dm = V.dofmap()

        subd_dofs = np.unique(np.hstack(
            [dm.cell_dofs(c.index())
             for c in f.SubsetIterator(volume_markers, self.volume)]))

        return np.max(self.function.vector().get_local()[subd_dofs])


class TotalVolume(DerivedQuantity):
    def __init__(self, field, volume) -> None:
        super().__init__(field)
        self.volume = volume
        self.title = "Total {} volume {}".format(self.field, self.volume)

    def compute(self):
        return f.assemble(self.function*self.dx(self.volume))


class TotalSurface(DerivedQuantity):
    def __init__(self, field, surface) -> None:
        super().__init__(field)
        self.surface = surface
        self.title = "Total {} surface {}".format(self.field, self.surface)

    def compute(self):
        return f.assemble(self.function*self.ds(self.surface))


class DerivedQuantities:
    def __init__(self, file=None, folder=None, nb_iterations_between_compute=1, nb_iterations_between_exports=None, **derived_quantities) -> None:
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
                            quantity_class(field=prms_dict["field"], volume=entity))
                if "surfaces" in prms_dict:
                    for entity in prms_dict["surfaces"]:
                        self.derived_quantities.append(
                            quantity_class(field=prms_dict["field"], surface=entity))

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

    def assign_properties_to_quantities(self, D, S, thermal_cond, H, T):
        for quantity in self.derived_quantities:
            quantity.D = D
            quantity.S = S
            quantity.thermal_cond = thermal_cond
            quantity.H = H
            quantity.T = T

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
            file_export = ''
            if self.folder is not None:
                file_export += self.folder + '/'
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
                        writer = csv.writer(f, lineterminator='\n')
                        for val in self.data:
                            writer.writerows([val])
                except OSError as err:
                    print("OS error: {0}".format(err))
                    print("The file " + file_export + ".txt might currently be busy."
                          "Please close the application then press any key.")
                    input()
        return True
