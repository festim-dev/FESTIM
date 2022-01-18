from FESTIM import DerivedQuantities, SurfaceFlux, AverageVolume, \
    TotalSurface, TotalVolume, MaximumVolume, MinimumVolume
import fenics as f
import os
from pathlib import Path


class TestMakeHeader:
    my_derv_quant = DerivedQuantities()
    surface_flux_1 = SurfaceFlux("solute", 2)
    surface_flux_2 = SurfaceFlux("T", 3)
    average_vol_1 = AverageVolume("solute", 3)
    average_vol_2 = AverageVolume("T", 4)
    tot_surf_1 = TotalSurface("retention", 6)
    tot_surf_2 = TotalSurface("trap1", 7)
    tot_vol_1 = TotalVolume("trap2", 5)
    min_vol_1 = MinimumVolume("retention", 2)
    max_vol_1 = MaximumVolume("T", 2)

    def test_simple(self):
        self.my_derv_quant.derived_quantities = [
            self.surface_flux_1
        ]
        header = self.my_derv_quant.make_header()
        expected_header = ["t(s)", self.surface_flux_1.title]
        assert header == expected_header

    def test_two_quantities(self):
        self.my_derv_quant.derived_quantities = [
            self.surface_flux_1,
            self.tot_surf_1,
        ]
        header = self.my_derv_quant.make_header()
        expected_header = ["t(s)", self.surface_flux_1.title, self.tot_surf_1.title]
        assert header == expected_header

    def test_all_quantities(self):
        self.my_derv_quant.derived_quantities = [
            self.surface_flux_1,
            self.average_vol_1,
            self.tot_surf_1,
            self.tot_vol_1,
            self.min_vol_1,
            self.max_vol_1,
        ]
        header = self.my_derv_quant.make_header()
        expected_header = ["t(s)"] + [
            self.surface_flux_1.title,
            self.average_vol_1.title,
            self.tot_surf_1.title,
            self.tot_vol_1.title,
            self.min_vol_1.title,
            self.max_vol_1.title,
        ]
        assert header == expected_header


class TestAssignMeasuresToQuantities:
    my_quantities = DerivedQuantities()
    my_quantities.derived_quantities = [
        SurfaceFlux("solute", 2),
        SurfaceFlux("T", 3),
        AverageVolume("solute", 3),
    ]
    mesh = f.UnitIntervalMesh(10)
    vol_markers = f.MeshFunction("size_t", mesh, 1)
    dx = f.dx(subdomain_data=vol_markers)
    ds = f.ds()
    n = f.FacetNormal(mesh)
    my_quantities.assign_measures_to_quantities(dx, ds)

    def test_quantities_have_dx(self):
        for quantity in self.my_quantities.derived_quantities:
            assert quantity.dx == self.dx

    def test_quantities_have_ds(self):
        for quantity in self.my_quantities.derived_quantities:
            assert quantity.ds == self.ds

    def test_quantities_have_n(self):
        for quantity in self.my_quantities.derived_quantities:
            assert quantity.n == self.n


def test_assign_function_to_quantities():
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    T = f.Function(V)
    solute = f.Function(V)
    my_quantities = DerivedQuantities()
    my_quantities.derived_quantities = [
        SurfaceFlux("solute", 2),
        SurfaceFlux("T", 3),
        AverageVolume("solute", 3),
    ]
    label_to_function = {
        "solute": solute,
        "T": T
    }

    my_quantities.assign_functions_to_quantities(label_to_function)

    for quantity in my_quantities.derived_quantities:
        assert quantity.solution == label_to_function[quantity.field]


class TestAssignPropertiesToQuantities:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_quantities = DerivedQuantities()
    my_quantities.derived_quantities = [
        SurfaceFlux("solute", 2),
        SurfaceFlux("T", 3),
        AverageVolume("solute", 3),
    ]
    D = f.Function(V)
    S = f.Function(V)
    H = f.Function(V)
    thermal_cond = f.Function(V)
    T = f.Function(V)

    my_quantities.assign_properties_to_quantities(D, S, thermal_cond, H, T)

    def test_quantities_have_D(self):
        for quantity in self.my_quantities.derived_quantities:
            assert quantity.D == self.D

    def test_quantities_have_S(self):
        for quantity in self.my_quantities.derived_quantities:
            assert quantity.S == self.S

    def test_quantities_have_H(self):
        for quantity in self.my_quantities.derived_quantities:
            assert quantity.H == self.H

    def test_quantities_have_thermal_cond(self):
        for quantity in self.my_quantities.derived_quantities:
            assert quantity.thermal_cond == self.thermal_cond


class TestCompute:
    my_derv_quant = DerivedQuantities()
    surface_flux_1 = SurfaceFlux("solute", 2)
    surface_flux_2 = SurfaceFlux("T", 3)
    average_vol_1 = AverageVolume("solute", 1)
    average_vol_2 = AverageVolume("T", 1)
    tot_surf_1 = TotalSurface("retention", 6)
    tot_surf_2 = TotalSurface("trap1", 7)
    tot_vol_1 = TotalVolume("trap1", 5)
    min_vol_1 = MinimumVolume("retention", 1)
    max_vol_1 = MaximumVolume("T", 1)

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    T = f.Function(V)
    label_to_function = {
        "solute": f.Function(V),
        "T": T,
        "retention": f.Function(V),
        "trap1": f.Function(V)
    }

    vol_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left = f.CompiledSubDomain("near(x[0], 0) && on_boundary")
    right = f.CompiledSubDomain("near(x[0], 1) && on_boundary")
    surface_markers = f.MeshFunction("size_t", mesh, 0)
    left.mark(surface_markers, 1)
    right.mark(surface_markers, 2)
    dx = f.Measure('dx', domain=mesh, subdomain_data=vol_markers)
    ds = f.Measure('ds', domain=mesh, subdomain_data=surface_markers)
    n = f.FacetNormal(mesh)

    D = f.interpolate(f.Constant(2), V)
    S = f.interpolate(f.Constant(2), V)
    H = f.interpolate(f.Constant(2), V)
    thermal_cond = f.interpolate(f.Constant(2), V)
    T = f.interpolate(f.Constant(2), V)

    def test_simple(self):
        self.my_derv_quant.derived_quantities = [
            self.surface_flux_1
        ]
        self.my_derv_quant.assign_functions_to_quantities(self.label_to_function)
        self.my_derv_quant.assign_properties_to_quantities(self.D, self.S, self.thermal_cond, self.H, self.T)
        self.my_derv_quant.assign_measures_to_quantities(self.dx, self.ds)
        t = 2

        expected_data = [t] + [quantity.compute() for quantity in self.my_derv_quant.derived_quantities]

        self.my_derv_quant.data = []
        self.my_derv_quant.compute(t, self.label_to_function)

        assert self.my_derv_quant.data[0] == expected_data

    def test_two_quantities(self):
        self.my_derv_quant.derived_quantities = [
            self.surface_flux_1,
            self.average_vol_1
        ]
        self.my_derv_quant.assign_functions_to_quantities(self.label_to_function)
        self.my_derv_quant.assign_properties_to_quantities(self.D, self.S, self.thermal_cond, self.H, self.T)
        self.my_derv_quant.assign_measures_to_quantities(self.dx, self.ds)
        t = 2

        expected_data = [t] + [quantity.compute() for quantity in self.my_derv_quant.derived_quantities]

        self.my_derv_quant.data = []
        self.my_derv_quant.compute(t, self.label_to_function)

        assert self.my_derv_quant.data[0] == expected_data

    def test_all_quantities(self):
        self.my_derv_quant.derived_quantities = [
            self.surface_flux_1,
            self.average_vol_1,
            self.tot_surf_1,
            self.tot_vol_1,
            self.min_vol_1,
            self.max_vol_1,
        ]
        self.my_derv_quant.assign_functions_to_quantities(self.label_to_function)
        self.my_derv_quant.assign_properties_to_quantities(self.D, self.S, self.thermal_cond, self.H, self.T)
        self.my_derv_quant.assign_measures_to_quantities(self.dx, self.ds)
        t = 2

        expected_data = [t]
        for quantity in self.my_derv_quant.derived_quantities:
            if isinstance(quantity, (MaximumVolume, MinimumVolume)):
                expected_data.append(quantity.compute(self.vol_markers))
            else:
                expected_data.append(quantity.compute())

        self.my_derv_quant.data = []
        self.my_derv_quant.compute(t, self.label_to_function)

        assert self.my_derv_quant.data[0] == expected_data


def test_write(tmpdir):
    """adds data to DerivedQuantities and checks that write() creates the csv
    file
    """
    file = "my_file"
    folder = str(Path(tmpdir.mkdir("test_folder")))
    my_derv_quant = DerivedQuantities(file=file, folder=folder)
    my_derv_quant.data = [
        ["a", "b", "c"],
        [1, 2, 3],
        [1, 2, 3],
    ]
    my_derv_quant.write()

    assert os.path.exists(folder + "/" + file + ".csv")
