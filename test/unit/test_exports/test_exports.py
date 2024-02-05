import festim
import pytest


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
