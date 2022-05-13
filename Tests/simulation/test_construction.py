import FESTIM as F
import pytest


def test_setting_traps():
    """Checks traps can be set with the expected types (F.Trap, list, or F.Traps)"""
    my_sim = F.Simulation()
    my_mat = F.Material(1, 1, 0)
    trap1 = F.Trap(1, 1, 1, 1, [my_mat], density=1)
    trap2 = F.Trap(2, 2, 2, 2, [my_mat], density=1)

    combinations = [trap1, [trap1], [trap1, trap2], F.Traps([trap1, trap2])]

    for trap_combination in combinations:
        my_sim.traps = trap_combination


def test_setting_traps_wrong_type():
    """Checks an error is raised when traps is set with the wrong type"""
    my_sim = F.Simulation()

    combinations = ["coucou", True]

    for trap_combination in combinations:
        with pytest.raises(
            TypeError,
            match="Accepted types for traps are list, FESTIM.Traps or FESTIM.Trap",
        ):
            my_sim.traps = trap_combination
