import festim
import pytest
import numpy as np


class TestExportsMethods:
    """Checks that festim.Exports methods work properly"""

    my_exp1 = festim.Export(field=0)
    my_exp2 = festim.Export(field="T")

    my_exports = festim.Exports([my_exp1])

    def test_exports_append(self):
        self.my_exports.append(self.my_exp2)
        assert self.my_exports == [self.my_exp1, self.my_exp2]

    def test_exports_insert(self):
        self.my_exports.insert(0, self.my_exp2)
        assert self.my_exports == [self.my_exp2, self.my_exp1, self.my_exp2]

    def test_exports_setitem(self):
        self.my_exports[0] = self.my_exp1
        assert self.my_exports == [self.my_exp1, self.my_exp1, self.my_exp2]

    def test_exports_extend_list_type(self):
        self.my_exports.extend([self.my_exp1])
        assert self.my_exports == [
            self.my_exp1,
            self.my_exp1,
            self.my_exp2,
            self.my_exp1,
        ]

    def test_exports_extend_self_type(self):
        self.my_exports.extend(festim.Exports([self.my_exp2]))
        assert self.my_exports == festim.Exports(
            [self.my_exp1, self.my_exp1, self.my_exp2, self.my_exp1, self.my_exp2]
        )


def test_set_exports_wrong_type():
    """Checks an error is raised when festim.Exports is set with the wrong type"""
    my_export = festim.Export(field=0)

    combinations = [my_export, "coucou", 1, True]

    for export_combination in combinations:
        with pytest.raises(
            TypeError,
            match="festim.Exports must be a list",
        ):
            festim.Exports(export_combination)

    with pytest.raises(
        TypeError,
        match="festim.Exports must be a list of festim.Export",
    ):
        festim.Exports([my_export, 2])


def test_assign_exports_wrong_type():
    """Checks an error is raised when the wrong type is assigned to festim.Exports"""
    my_export = festim.Export(field=0)
    my_exports = festim.Exports([my_export])

    combinations = ["coucou", 1, True]

    error_pattern = "festim.Exports must be a list of festim.Export"

    for export_combination in combinations:
        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_exports.append(export_combination)

        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_exports.extend([export_combination])

        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_exports[0] = export_combination

        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_exports.insert(0, export_combination)


class TestExportsPropertyDeprWarn:
    """
    A temporary test to check DeprecationWarnings in festim.Exports.exports
    """

    my_export = festim.Export(field=0)
    my_exports = festim.Exports([])

    def test_property_depr_warns(self):
        with pytest.deprecated_call():
            self.my_exports.exports

    def test_property_setter_depr_warns(self):
        with pytest.deprecated_call():
            self.my_exports.exports = [self.my_export]


class TestExportsPropertyRaiseError:
    """
    A temporary test to check TypeErrors in festim.Exports.exports
    """

    my_export = festim.Export(field=0)
    my_exports = festim.Exports([])

    def test_set_exports_wrong_type(self):
        with pytest.raises(
            TypeError,
            match="exports must be a list",
        ):
            self.my_exports.exports = self.my_export

    def test_set_exports_list_wrong_type(self):
        with pytest.raises(
            TypeError,
            match="exports must be a list of festim.Export",
        ):
            self.my_exports.exports = [self.my_export, 1]


def test_instanciate_with_no_elements():
    """
    Test to catch bug described in issue #724
    """
    # define exports
    festim.Exports()


def test_show_units_true_with_retention():
    """Test to catch the bug in issue #765"""
    my_model = festim.Simulation()
    my_model.mesh = festim.MeshFromVertices(np.linspace(0, 1, num=200))

    my_model.materials = festim.Material(id=1, D_0=1, E_D=0)

    my_model.traps = festim.Trap(
        k_0=1,
        E_k=0,
        p_0=1,
        E_p=0,
        density=1.3e-3,
        materials=my_model.materials[0],
    )

    my_model.T = festim.Temperature(500)

    my_model.settings = festim.Settings(
        absolute_tolerance=1e10, relative_tolerance=1e-09, transient=False
    )

    my_model.exports = [
        festim.DerivedQuantities(
            [
                festim.TotalVolume("retention", volume=1),
                festim.TotalSurface("retention", surface=1),
            ],
            show_units=True,
        )
    ]

    my_model.initialise()
    my_model.run()
