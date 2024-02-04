import festim
import pytest


def test_set_traps_wrong_type():
    """Checks an error is raised when festim.Exports is set with the wrong type"""
    export = festim.Export(field=0)

    combinations = [export, "coucou", 1, True]

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
        festim.Exports([export, 2])
