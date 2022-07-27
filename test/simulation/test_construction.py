import festim as F
import pytest


def test_setting_traps():
    """Checks traps can be set with the expected types (F.Trap, list, or F.Traps)"""
    my_sim = F.Simulation()
    my_mat = F.Material(1, 1, 0)
    trap1 = F.Trap(1, 1, 1, 1, [my_mat], density=1)
    trap2 = F.Trap(2, 2, 2, 2, [my_mat], density=1)

    combinations = [trap1, [trap1], [trap1, trap2], F.Traps([trap1, trap2])]

    for combination in combinations:
        my_sim.traps = combination


def test_setting_traps_wrong_type():
    """Checks an error is raised when traps is set with the wrong type"""
    my_sim = F.Simulation()

    combinations = ["coucou", True]

    for combination in combinations:
        with pytest.raises(
            TypeError,
            match="Accepted types for traps are list, festim.Traps or festim.Trap",
        ):
            my_sim.traps = combination


def test_setting_materials():
    """Checks materials can be set with the expected types (F.Material, list, or F.Materials)"""
    my_sim = F.Simulation()
    mat1 = F.Material(1, 1, 1)
    mat2 = F.Material(2, 2, 2)

    combinations = [mat1, [mat1], [mat1, mat2], F.Materials([mat1, mat2])]

    for combination in combinations:
        my_sim.materials = combination


def test_setting_materials_wrong_type():
    """Checks an error is raised when materials is set with the wrong type"""
    my_sim = F.Simulation()

    combinations = ["coucou", True]

    for combination in combinations:
        with pytest.raises(
            TypeError,
            match="accepted types for materials are list, festim.Material or festim.Materials",
        ):
            my_sim.materials = combination


def test_setting_exports():
    """Checks exports can be set with the expected types (F.Material, list, or F.Exports)"""
    my_sim = F.Simulation()
    export1 = F.XDMFExport("solute")
    export2 = F.XDMFExport("solute")

    combinations = [
        export1,
        [export1],
        [export1, export2],
        F.Exports([export1, export2]),
    ]

    for trap_combination in combinations:
        my_sim.exports = trap_combination


def test_setting_exports_wrong_type():
    """Checks an error is raised when exports is set with the wrong type"""
    my_sim = F.Simulation()

    combinations = ["coucou", True]

    for trap_combination in combinations:
        with pytest.raises(
            TypeError,
            match="accepted types for exports are list, festim.Export or festim.Exports",
        ):
            my_sim.exports = trap_combination
