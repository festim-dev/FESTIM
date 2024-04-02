import festim as F
import pytest


class TestSettingTrapsMaterialsExprots:
    my_sim = F.Simulation()
    mat1 = F.Material(1, 1, 1)
    mat2 = F.Material(2, 2, 2)
    trap1 = F.Trap(1, 1, 1, 1, [mat1], density=1)
    trap2 = F.Trap(2, 2, 2, 2, [mat1], density=1)
    export1 = F.XDMFExport("solute")
    export2 = F.XDMFExport("solute")
    wrong_types = ["coucou", True]

    @pytest.mark.parametrize(
        "trap", [trap1, [trap1], [trap1, trap2], F.Traps([trap1, trap2])]
    )
    def test_setting_traps(self, trap):
        """Checks traps can be set with the expected types (F.Trap, list, or F.Traps)"""

        self.my_sim.traps = trap

    @pytest.mark.parametrize("trap", wrong_types)
    def test_setting_traps_wrong_type(self, trap):
        """Checks an error is raised when traps is set with the wrong type"""

        with pytest.raises(
            TypeError,
            match="Accepted types for traps are list, festim.Traps or festim.Trap",
        ):
            self.my_sim.traps = trap

    @pytest.mark.parametrize(
        "mat", [mat1, [mat1], [mat1, mat2], F.Materials([mat1, mat2])]
    )
    def test_setting_materials(self, mat):
        """Checks materials can be set with the expected types (F.Material, list, or F.Materials)"""

        self.my_sim.materials = mat

    @pytest.mark.parametrize("mat", wrong_types)
    def test_setting_materials_wrong_type(self, mat):
        """Checks an error is raised when materials is set with the wrong type"""

        with pytest.raises(
            TypeError,
            match="accepted types for materials are list, festim.Material or festim.Materials",
        ):
            self.my_sim.materials = mat

    @pytest.mark.parametrize(
        "exp", [export1, [export1], [export1, export2], F.Exports([export1, export2])]
    )
    def test_setting_exports(self, exp):
        """Checks exports can be set with the expected types (F.Material, list, or F.Exports)"""

        self.my_sim.exports = exp

    @pytest.mark.parametrize("exp", wrong_types)
    def test_setting_exports_wrong_type(self, exp):
        """Checks an error is raised when exports is set with the wrong type"""

        with pytest.raises(
            TypeError,
            match="accepted types for exports are list, festim.Export or festim.Exports",
        ):
            self.my_sim.exports = exp
