from festim import (
    DerivedQuantities,
    SurfaceFlux,
    AverageVolume,
    TotalSurface,
    TotalVolume,
    MaximumVolume,
    MinimumVolume,
    MinimumSurface,
    MaximumSurface,
    AverageSurface,
    PointValue,
    SurfaceFluxCylindrical,
    SurfaceFluxSpherical,
    AdsorbedHydrogen,
    Materials,
)
import fenics as f
import os
from pathlib import Path
import pytest


class TestMakeHeader:
    surface_flux_1 = SurfaceFlux("solute", 2)
    surface_flux_2 = SurfaceFlux("T", 3)
    average_vol_1 = AverageVolume("solute", 3)
    average_vol_2 = AverageVolume("T", 4)
    tot_surf_1 = TotalSurface("retention", 6)
    tot_surf_2 = TotalSurface("trap1", 7)
    tot_vol_1 = TotalVolume("trap2", 5)
    min_vol_1 = MinimumVolume("retention", 2)
    min_vol_2 = MinimumVolume("T", 2)
    max_vol_1 = MaximumVolume("T", 2)
    max_vol_2 = MaximumVolume("trap2", 2)
    min_surface_1 = MinimumSurface("solute", 1)
    min_surface_2 = MinimumSurface("T", 2)
    max_surface_1 = MaximumSurface("solute", 8)
    max_surface_2 = MaximumSurface("T", 9)
    avg_surface_1 = AverageSurface("solute", 4)
    avg_surface_2 = AverageSurface("T", 6)
    point_1 = PointValue(field="retention", x=2)
    point_2 = PointValue(field="T", x=9)
    cyl_surface_flux_1 = SurfaceFluxCylindrical("solute", 2)
    cyl_surface_flux_2 = SurfaceFluxCylindrical("T", 3)
    sph_surface_flux_1 = SurfaceFluxSpherical("solute", 5)
    sph_surface_flux_2 = SurfaceFluxSpherical("T", 6)
    ads_h = AdsorbedHydrogen(1)

    def test_simple(self):
        """
        Tests that a proper header is made for the output .csv file
        when there is only one festim.DerivedQuantity object
        """
        my_derv_quant = DerivedQuantities([self.surface_flux_1])
        header = my_derv_quant.make_header()
        expected_header = ["t(s)", self.surface_flux_1.title]
        assert header == expected_header

    def test_two_quantities(self):
        """
        Tests that a proper header is made for the output .csv file
        when there are two festim.DerivedQuantity objects
        """
        my_derv_quant = DerivedQuantities(
            [
                self.surface_flux_1,
                self.tot_surf_1,
            ]
        )
        header = my_derv_quant.make_header()
        expected_header = ["t(s)", self.surface_flux_1.title, self.tot_surf_1.title]
        assert header == expected_header

    def test_all_quantities(self):
        """
        Tests that a proper header is made for the output .csv file
        when there are many festim.DerivedQuantity objects
        """
        my_derv_quant = DerivedQuantities(
            [
                self.surface_flux_1,
                self.average_vol_1,
                self.tot_surf_1,
                self.tot_vol_1,
                self.min_vol_1,
                self.max_vol_1,
                self.point_1,
                self.ads_h,
            ]
        )
        header = my_derv_quant.make_header()
        expected_header = ["t(s)"] + [
            self.surface_flux_1.title,
            self.average_vol_1.title,
            self.tot_surf_1.title,
            self.tot_vol_1.title,
            self.min_vol_1.title,
            self.max_vol_1.title,
            self.point_1.title,
            self.ads_h.title,
        ]
        assert header == expected_header

    def test_with_units_simple(self):
        """Test with quantities that don't require mesh dimension for unit"""
        my_derv_quant = DerivedQuantities(
            [
                self.average_vol_1,
                self.average_vol_2,
                self.min_vol_1,
                self.min_vol_2,
                self.max_vol_1,
                self.max_vol_2,
                self.min_surface_1,
                self.min_surface_2,
                self.max_surface_1,
                self.max_surface_2,
                self.avg_surface_1,
                self.avg_surface_2,
                self.point_1,
                self.point_2,
                self.cyl_surface_flux_1,
                self.cyl_surface_flux_2,
                self.sph_surface_flux_1,
                self.sph_surface_flux_2,
                self.ads_h,
            ],
            show_units=True,
        )
        header = my_derv_quant.make_header()
        expected_header = ["t(s)"] + [
            "Average solute volume 3 (H m-3)",
            "Average T volume 4 (K)",
            "Minimum retention volume 2 (H m-3)",
            "Minimum T volume 2 (K)",
            "Maximum T volume 2 (K)",
            "Maximum trap2 volume 2 (H m-3)",
            "Minimum solute surface 1 (H m-3)",
            "Minimum T surface 2 (K)",
            "Maximum solute surface 8 (H m-3)",
            "Maximum T surface 9 (K)",
            "Average solute surface 4 (H m-3)",
            "Average T surface 6 (K)",
            "retention value at [2] (H m-3)",
            "T value at [9] (K)",
            "solute flux surface 2 (H s-1)",
            "Heat flux surface 3 (W)",
            "solute flux surface 5 (H s-1)",
            "Heat flux surface 6 (W)",
            "Adsorbed H on surface 1 (H m-2)",
        ]
        assert header == expected_header


class TestAssignMeasuresToQuantities:
    """
    Tests that measure attributes are properly assigned to all
    festim.DerivedQuantity objects
    """

    my_quantities = DerivedQuantities(
        [
            SurfaceFlux("solute", 2),
            SurfaceFlux("T", 3),
            AverageVolume("solute", 3),
        ]
    )
    mesh = f.UnitIntervalMesh(10)
    vol_markers = f.MeshFunction("size_t", mesh, 1)
    dx = f.dx(subdomain_data=vol_markers)
    ds = f.ds()
    n = f.FacetNormal(mesh)
    my_quantities.assign_measures_to_quantities(dx, ds)

    def test_quantities_have_dx(self):
        """Check for the volume measure"""
        for quantity in self.my_quantities:
            assert quantity.dx == self.dx

    def test_quantities_have_ds(self):
        """Check for the surface measure"""
        for quantity in self.my_quantities:
            assert quantity.ds == self.ds

    def test_quantities_have_n(self):
        """Check for the normal vector of the surface"""
        for quantity in self.my_quantities:
            assert quantity.n == self.n


class TestAssignPropertiesToQuantities:
    """
    Tests that property attributes are properly assigned to all
    festim.DerivedQuantity objects
    """

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    my_quantities = DerivedQuantities(
        [
            SurfaceFlux("solute", 2),
            SurfaceFlux("T", 3),
            AverageVolume("solute", 3),
        ]
    )
    my_mats = Materials([])
    my_mats.D = f.Function(V)
    my_mats.S = f.Function(V)
    my_mats.Q = f.Function(V)
    my_mats.thermal_cond = f.Function(V)
    T = f.Function(V)

    my_quantities.assign_properties_to_quantities(my_mats)

    def test_quantities_have_D(self):
        """Check for diffusivity"""
        for quantity in self.my_quantities:
            assert quantity.D == self.my_mats.D

    def test_quantities_have_S(self):
        """Check for solubility"""
        for quantity in self.my_quantities:
            assert quantity.S == self.my_mats.S

    def test_quantities_have_Q(self):
        """Check for Soret"""
        for quantity in self.my_quantities:
            assert quantity.Q == self.my_mats.Q

    def test_quantities_have_thermal_cond(self):
        """Check for thermal conductivity"""
        for quantity in self.my_quantities:
            assert quantity.thermal_cond == self.my_mats.thermal_cond


class TestCompute:
    """Test that the derived qunatities compute the correct value"""

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
        "trap1": f.Function(V),
    }

    vol_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left = f.CompiledSubDomain("near(x[0], 0) && on_boundary")
    right = f.CompiledSubDomain("near(x[0], 1) && on_boundary")
    surface_markers = f.MeshFunction("size_t", mesh, 0)
    left.mark(surface_markers, 1)
    right.mark(surface_markers, 2)
    dx = f.Measure("dx", domain=mesh, subdomain_data=vol_markers)
    ds = f.Measure("ds", domain=mesh, subdomain_data=surface_markers)
    n = f.FacetNormal(mesh)

    T = f.interpolate(f.Constant(2), V)
    my_mats = Materials([])
    my_mats.D = f.interpolate(f.Constant(2), V)
    my_mats.S = f.interpolate(f.Constant(2), V)
    my_mats.H = f.interpolate(f.Constant(2), V)
    my_mats.thermal_cond = f.interpolate(f.Constant(2), V)

    def test_simple(self):
        """Check for the case of one festim.DerivedQuantity object"""
        my_derv_quant = DerivedQuantities([self.surface_flux_1])
        for quantity in my_derv_quant:
            quantity.function = self.label_to_function[quantity.field]
        my_derv_quant.assign_properties_to_quantities(self.my_mats)
        my_derv_quant.assign_measures_to_quantities(self.dx, self.ds)
        t = 2

        expected_data = [t] + [quantity.compute() for quantity in my_derv_quant]

        my_derv_quant.data = []
        my_derv_quant.compute(t)

        # title created in compute() method and appended to data
        # so test line 2 for first data entry

        assert my_derv_quant.data[1] == expected_data

    def test_two_quantities(self):
        """Check for the case of two festim.DerivedQuantity objects"""
        my_derv_quant = DerivedQuantities(
            [
                self.surface_flux_1,
                self.average_vol_1,
            ]
        )
        for quantity in my_derv_quant:
            quantity.function = self.label_to_function[quantity.field]
        my_derv_quant.assign_properties_to_quantities(self.my_mats)
        my_derv_quant.assign_measures_to_quantities(self.dx, self.ds)
        t = 2

        expected_data = [t] + [quantity.compute() for quantity in my_derv_quant]

        my_derv_quant.data = []
        my_derv_quant.compute(t)

        # title created in compute() method and appended to data
        # so test line 2 for first data entry

        assert my_derv_quant.data[1] == expected_data

    def test_all_quantities(self):
        """Check for the case of many festim.DerivedQuantity objects"""
        my_derv_quant = DerivedQuantities(
            [
                self.surface_flux_1,
                self.average_vol_1,
                self.tot_surf_1,
                self.tot_vol_1,
                self.min_vol_1,
                self.max_vol_1,
            ]
        )
        for quantity in my_derv_quant:
            quantity.function = self.label_to_function[quantity.field]
        my_derv_quant.assign_properties_to_quantities(self.my_mats)
        my_derv_quant.assign_measures_to_quantities(self.dx, self.ds)
        t = 2

        expected_data = [t]
        for quantity in my_derv_quant:
            if isinstance(quantity, (MaximumVolume, MinimumVolume)):
                expected_data.append(quantity.compute(self.vol_markers))
            else:
                expected_data.append(quantity.compute())

        my_derv_quant.data = []
        my_derv_quant.compute(t)

        # title created in compute() method and appended to data
        # so test line 2 for first data entry

        assert my_derv_quant.data[1] == expected_data


class TestWrite:
    @pytest.fixture
    def folder(self, tmpdir):
        return str(Path(tmpdir.mkdir("test_folder")))

    @pytest.fixture
    def my_derived_quantities(self):
        filename = "my_file.csv"
        my_derv_quant = DerivedQuantities([], filename=filename)
        my_derv_quant.data = [
            ["a", "b", "c"],
            [1, 2, 3],
            [1, 2, 3],
        ]
        return my_derv_quant

    def test_write(self, folder, my_derived_quantities):
        """adds data to DerivedQuantities and checks that write() creates the csv
        file
        """
        filename = "{}/my_file.csv".format(folder)
        my_derived_quantities.filename = filename
        my_derived_quantities.write()

        assert os.path.exists(filename)

    def test_write_folder_doesnt_exist(self, folder, my_derived_quantities):
        """Checks that write() creates the inexisting folder"""
        filename = "{}/folder2/my_file.csv".format(folder)
        my_derived_quantities.filename = filename
        my_derived_quantities.write()

        assert os.path.exists(filename)


class TestFilter:
    """Tests the filter method of DerivedQUantities"""

    def test_simple(self):
        flux1 = SurfaceFlux(field="solute", surface=1)
        flux2 = SurfaceFlux(field="T", surface=2)
        derived_quantities = DerivedQuantities([flux1, flux2])

        assert derived_quantities.filter(surfaces=[1, 2]) == [flux1, flux2]
        assert derived_quantities.filter(surfaces=[1]) == flux1
        assert derived_quantities.filter(fields=["T"]) == flux2
        assert derived_quantities.filter(fields=["T", "solute"], surfaces=[3]) == []
        assert derived_quantities.filter(fields=["solute"], surfaces=[1, 2]) == flux1

    def test_with_volumes(self):
        flux1 = SurfaceFlux(field="solute", surface=1)
        flux2 = SurfaceFlux(field="T", surface=2)
        total1 = TotalVolume(field="1", volume=3)
        total2 = TotalVolume(field="retention", volume=1)
        derived_quantities = DerivedQuantities([flux1, flux2, total1, total2])

        assert derived_quantities.filter() == derived_quantities
        assert derived_quantities.filter(surfaces=[1, 2], volumes=[3]) == []
        assert (
            derived_quantities.filter(volumes=[1, 3], fields=["retention", "solute"])
            == total2
        )

    def test_with_single_args(self):
        flux1 = SurfaceFlux(field="solute", surface=1)
        flux2 = SurfaceFlux(field="T", surface=2)
        total1 = TotalVolume(field="1", volume=3)

        derived_quantities = DerivedQuantities([flux1, flux2, total1])

        assert derived_quantities.filter(surfaces=1) == flux1
        assert derived_quantities.filter(fields="T") == flux2
        assert derived_quantities.filter(volumes=3) == total1

    def test_several_quantities_one_surface(self):
        surf1 = SurfaceFlux(field="solute", surface=1)
        surf2 = TotalSurface(field="solute", surface=1)
        derived_quantities = DerivedQuantities([surf1, surf2])

        assert derived_quantities.filter(surfaces=1, instances=SurfaceFlux) == surf1
        assert derived_quantities.filter(surfaces=1, instances=TotalSurface) == surf2
        assert derived_quantities.filter(
            surfaces=1, instances=[TotalSurface, SurfaceFlux]
        ) == [surf1, surf2]


def test_wrong_type_filename():
    """Checks that an error is raised when filename is not a string"""
    with pytest.raises(TypeError, match="filename must be a string"):
        DerivedQuantities([], filename=2)


def test_filename_ends_with_csv():
    """Checks that an error is raised when filename doesn't end with .csv"""
    with pytest.raises(ValueError, match="filename must end with .csv"):
        DerivedQuantities([], filename="coucou")


class TestDerivedQuantititesMethods:
    """Checks that festim.DerivedQuantitites methods work properly"""

    flux1 = SurfaceFlux(field="solute", surface=1)
    flux2 = SurfaceFlux(field="solute", surface=2)

    my_dqs = DerivedQuantities([flux1])

    def test_DQs_append(self):
        self.my_dqs.append(self.flux2)
        assert self.my_dqs == [self.flux1, self.flux2]

    def test_DQs_insert(self):
        self.my_dqs.insert(0, self.flux2)
        assert self.my_dqs == [self.flux2, self.flux1, self.flux2]

    def test_DQs_setitem(self):
        self.my_dqs[0] = self.flux1
        assert self.my_dqs == [self.flux1, self.flux1, self.flux2]

    def test_DQs_extend_list_type(self):
        self.my_dqs.extend([self.flux1])
        assert self.my_dqs == [self.flux1, self.flux1, self.flux2, self.flux1]

    def test_DQs_extend_self_type(self):
        self.my_dqs.extend(DerivedQuantities([self.flux2]))
        assert self.my_dqs == DerivedQuantities(
            [self.flux1, self.flux1, self.flux2, self.flux1, self.flux2]
        )


def test_set_derived_quantitites_wrong_type():
    """Checks an error is raised when festim.DerivedQuantities is set with the wrong type"""
    flux1 = SurfaceFlux(field="solute", surface=1)

    combinations = [flux1, "coucou", 1, True]

    for dq_combination in combinations:
        with pytest.raises(
            TypeError,
            match="festim.DerivedQuantities must be a list",
        ):
            DerivedQuantities(dq_combination)

    with pytest.raises(
        TypeError,
        match="festim.DerivedQuantities must be a list of festim.DerivedQuantity",
    ):
        DerivedQuantities([flux1, 2])


def test_assign_derived_quantitites_wrong_type():
    """Checks an error is raised when the wrong type is assigned to festim.DerivedQuantities"""
    my_derived_quantities = DerivedQuantities([])
    combinations = ["coucou", 1, True]
    error_pattern = "festim.DerivedQuantities must be a list of festim.DerivedQuantity"

    for dq_combination in combinations:
        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_derived_quantities.append(dq_combination)

        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_derived_quantities.extend([dq_combination])

        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_derived_quantities[0] = dq_combination

        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_derived_quantities.insert(0, dq_combination)


class TestDerivedQuantititesPropertyDeprWarn:
    """
    A temporary test to check DeprecationWarnings in festim.DerivedQuantitites.exports
    """

    my_derived_quantity = SurfaceFlux(0, 2)
    my_derived_quantities = DerivedQuantities([])

    def test_property_depr_warns(self):
        with pytest.deprecated_call():
            self.my_derived_quantities.derived_quantities

    def test_property_setter_depr_warns(self):
        with pytest.deprecated_call():
            self.my_derived_quantities.derived_quantities = [self.my_derived_quantity]


class TestDerivedQuantititesPropertyRaiseError:
    """
    A temporary test to check TypeErrors in festim.DerivedQuantitites.exports
    """

    my_derived_quantity = SurfaceFlux(0, 2)
    my_derived_quantities = DerivedQuantities([])

    def test_set_der_quants_wrong_type(self):
        with pytest.raises(
            TypeError,
            match="derived_quantities must be a list",
        ):
            self.my_derived_quantities.derived_quantities = self.my_derived_quantity

    def test_set_der_quants_list_wrong_type(self):
        with pytest.raises(
            TypeError,
            match="derived_quantities must be a list of festim.DerivedQuantity",
        ):
            self.my_derived_quantities.derived_quantities = [
                self.my_derived_quantity,
                1,
            ]


def test_instanciate_with_no_derived_quantities():
    """
    Test to catch bug described in issue #724
    """
    # define exports
    folder_results = "Results/"
    DerivedQuantities(
        filename=folder_results + "derived_quantities.csv",
        nb_iterations_between_exports=1,
    )
