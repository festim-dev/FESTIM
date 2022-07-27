from festim import Material, Materials
import festim as F
from fenics import *
import pytest

from festim.temperature.temperature_solver import HeatTransferProblem


def test_find_material_from_id():
    """Tests the function find_material_from_id() for cases with one id per
    material
    """
    mat_1 = Material(id=1, D_0=None, E_D=None)
    mat_2 = Material(id=2, D_0=None, E_D=None)
    my_Mats = Materials([mat_1, mat_2])
    assert my_Mats.find_material_from_id(1) == mat_1
    assert my_Mats.find_material_from_id(2) == mat_2


def test_find_material_from_id_with_several_ids():
    """Tests the function find_material_from_id() for cases with several ids
    per material
    """

    mat_1 = Material(id=[1, 2], D_0=None, E_D=None)
    my_Mats = Materials([mat_1])
    assert my_Mats.find_material_from_id(1) == mat_1
    assert my_Mats.find_material_from_id(2) == mat_1


def test_find_material_from_id_unfound_id():
    """
    Tests the function find_material_from_id with a list of materials
    without the searched ID
        - check that an error is rasied
    """
    mat_1 = Material(id=5, D_0=None, E_D=None)
    mat_2 = Material(id=2, D_0=None, E_D=None)
    mat_3 = Material(id=-1, D_0=None, E_D=None)

    my_Mats = Materials([mat_1, mat_2, mat_3])
    id_test = 1
    with pytest.raises(ValueError, match="Couldn't find ID {}".format(id_test)):
        my_Mats.find_material_from_id(id_test)


def test_find_material_from_name():
    """Checks the function find_material_from_name() returns the correct material"""
    mat_1 = Material(id=1, D_0=None, E_D=None, name="mat1")
    mat_2 = Material(id=2, D_0=None, E_D=None, name="mat2")
    my_Mats = Materials([mat_1, mat_2])
    assert my_Mats.find_material_from_name("mat1") == mat_1
    assert my_Mats.find_material_from_name("mat2") == mat_2


def test_find_material_from_name_unfound_name():
    """
    Check find_material_from_name raises an error when the name is not found
    """
    mat_1 = Material(id=5, D_0=None, E_D=None)
    mat_2 = Material(id=2, D_0=None, E_D=None)
    mat_3 = Material(id=-1, D_0=None, E_D=None)

    my_Mats = Materials([mat_1, mat_2, mat_3])
    name_test = "coucou"
    with pytest.raises(
        ValueError, match="No material with name {} was found".format(name_test)
    ):
        my_Mats.find_material_from_name(name_test)


def test_unused_thermal_cond():
    """
    Checks warnings when some keys are unused
    """

    mat_1 = Material(id=1, D_0=1, E_D=2, thermal_cond=2)
    my_mats = Materials([mat_1])
    with pytest.warns(UserWarning, match=r"thermal_cond key will be ignored"):
        my_mats.check_for_unused_properties(T=F.Temperature(100), derived_quantities=[])

    # this shouldn't throw warnings
    derived_quantities = [F.SurfaceFlux("T", surface=0)]
    # record the warnings
    with pytest.warns(None) as record:
        my_mats.check_for_unused_properties(
            T=F.Temperature(100), derived_quantities=derived_quantities
        )

    # check that no warning were raised
    assert len(record) == 0


def test_missing_thermal_cond():
    """Tests that an error is raised when the thermal cond is missing"""
    my_mats = Materials([Material(1, D_0=1, E_D=1)])
    with pytest.raises(ValueError, match="Missing thermal_cond in materials"):
        my_mats.check_missing_properties(
            T=F.HeatTransferProblem(), derived_quantities=[]
        )


def test_missing_heat_capacity():
    """Tests that an error is raised when the heat_capacity is missing"""
    my_mats = Materials([Material(1, D_0=1, E_D=1, thermal_cond=1, rho=1)])
    with pytest.raises(ValueError, match="Missing heat_capacity in materials"):
        my_mats.check_missing_properties(
            T=F.HeatTransferProblem(), derived_quantities=[]
        )


def test_missing_rho():
    """Tests that an error is raised when the rho is missing"""
    my_mats = Materials([Material(1, D_0=1, E_D=1, thermal_cond=1, heat_capacity=1)])
    with pytest.raises(ValueError, match="Missing rho in materials"):
        my_mats.check_missing_properties(
            T=F.HeatTransferProblem(), derived_quantities=[]
        )


def test_different_ids_in_materials():
    """
    Checks that an error is raised when two materials have the same id
    """

    mat_1 = Material(id=1, D_0=1, E_D=2)
    mat_2 = Material(id=1, D_0=2, E_D=3)
    my_mats = Materials([mat_1, mat_2])
    with pytest.raises(ValueError, match=r"Some materials have the same id"):
        my_mats.check_unique_ids()


def test_unused_keys():
    """
    Checks warnings when some keys are unused
    """

    mat_1 = Material(id=1, D_0=1, E_D=2, rho=2)
    my_mats = Materials([mat_1])

    with pytest.warns(UserWarning, match=r"rho key will be ignored"):
        my_mats.check_for_unused_properties(T=F.Temperature(200), derived_quantities={})

    mat_1.rho = None
    mat_1.heat_capacity = 2

    with pytest.warns(UserWarning, match=r"heat_capacity key will be ignored"):
        my_mats.check_for_unused_properties(T=F.Temperature(200), derived_quantities={})


def test_non_matching_properties():
    mat_1 = Material(id=1, D_0=1, E_D=2, rho=2)
    mat_2 = Material(id=1, D_0=1, E_D=2)
    my_mats = Materials([mat_1, mat_2])
    with pytest.raises(ValueError, match=r"rho is not defined for all materials"):
        my_mats.check_consistency()


class TestCheckBorders:
    def test_works(self):
        materials = [
            Material(id=1, D_0=None, E_D=None, borders=[0.5, 0.7]),
            Material(id=2, D_0=None, E_D=None, borders=[0, 0.5]),
        ]
        size = 0.7
        assert Materials(materials).check_borders(size) is True

    def test_not_beginning_at_zero(self):
        with pytest.raises(ValueError, match=r"zero"):
            size = 0.7
            materials = [
                Material(id=1, D_0=None, E_D=None, borders=[0.5, 0.7]),
                Material(id=1, D_0=None, E_D=None, borders=[0.2, 0.5]),
            ]
            Materials(materials).check_borders(size)

    def test_not_matching(self):
        with pytest.raises(ValueError, match=r"each other"):
            materials = [
                Material(id=1, D_0=None, E_D=None, borders=[0.5, 1]),
                Material(id=1, D_0=None, E_D=None, borders=[0, 0.6]),
                Material(id=1, D_0=None, E_D=None, borders=[0.6, 1]),
            ]
            size = 1
            Materials(materials).check_borders(size)

    def test_not_matching_with_size(self):
        with pytest.raises(ValueError, match=r"size"):
            materials = [
                Material(id=1, D_0=None, E_D=None, borders=[0, 1]),
            ]
            size = 3
            Materials(materials).check_borders(size)

    def test_1_material_2_subdomains(self):
        materials = Materials([Material([1, 2], 1, 0, borders=[[0, 1], [1, 9]])])

        materials.check_borders(size=9)

    def test_2_materials_3_subdomains(self):
        materials = Materials(
            [
                Material([1, 2], 1, 0, borders=[[0, 1], [1, 5]]),
                Material(3, 1, 0, borders=[5, 9]),
            ]
        )
        materials.check_borders(size=9)

    def test_1_material_1_id_2_borders(self):
        materials = Materials(
            [
                Material(1, 1, 0, borders=[[0, 1], [1, 9]]),
            ]
        )
        materials.check_borders(size=9)


def test_material_with_multiple_ids_solubility():
    """Tests the function find_material_from_id() for cases with several ids
    per material
    """

    mat_1 = Material(id=[1, 2], D_0=1, E_D=1)
    my_mats = Materials([mat_1])
    mesh = UnitIntervalMesh(10)
    vm = MeshFunction("size_t", mesh, 1, 1)
    my_mats.create_properties(vm, T=Constant(300))
    V = FunctionSpace(mesh, "P", 1)
    interpolate(my_mats.D, V)


def test_create_properties():
    """
    Test the function create_properties()
    """
    mesh = UnitIntervalMesh(10)
    DG_1 = FunctionSpace(mesh, "DG", 1)
    mat_1 = Material(
        1,
        D_0=1,
        E_D=0,
        S_0=7,
        E_S=0,
        thermal_cond=4,
        heat_capacity=5,
        rho=6,
        H={"free_enthalpy": 5, "entropy": 6},
    )
    mat_2 = Material(
        2,
        D_0=2,
        E_D=0,
        S_0=8,
        E_S=0,
        thermal_cond=5,
        heat_capacity=6,
        rho=7,
        H={"free_enthalpy": 6, "entropy": 6},
    )
    materials = Materials([mat_1, mat_2])
    mf = MeshFunction("size_t", mesh, 1, 0)
    for cell in cells(mesh):
        x = cell.midpoint().x()
        if x < 0.5:
            mf[cell] = 1
        else:
            mf[cell] = 2
    T = Expression("1", degree=1)
    materials.create_properties(mf, T)
    D = interpolate(materials.D, DG_1)
    thermal_cond = interpolate(materials.thermal_cond, DG_1)
    cp = interpolate(materials.heat_capacity, DG_1)
    rho = interpolate(materials.density, DG_1)
    H = interpolate(materials.H, DG_1)
    S = interpolate(materials.S, DG_1)

    for cell in cells(mesh):
        assert D(cell.midpoint().x()) == mf[cell]
        assert thermal_cond(cell.midpoint().x()) == mf[cell] + 3
        assert cp(cell.midpoint().x()) == mf[cell] + 4
        assert rho(cell.midpoint().x()) == mf[cell] + 5
        assert H(cell.midpoint().x()) == mf[cell] + 10
        assert S(cell.midpoint().x()) == mf[cell] + 6


def test_E_S_without_S_0():
    with pytest.raises(ValueError, match="S_0 cannot be None"):
        Material(1, 1, 1, S_0=None, E_S=1)


def test_S_0_without_E_S():
    with pytest.raises(ValueError, match="E_S cannot be None"):
        Material(1, 1, 1, S_0=1, E_S=None)


def test_error_wrong_solubility_law_string():
    """Tests that an error is raised when the wrong value of solubility_law is set"""
    with pytest.raises(
        ValueError,
        match="Acceptable values for solubility_law are 'henry' and 'sievert'",
    ):
        Material(1, 1, 1, solubility_law="foo")
