import festim
import pytest


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
