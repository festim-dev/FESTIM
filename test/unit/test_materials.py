import festim as F
from fenics import *
import pytest
import warnings


def test_find_material_from_id():
    """Tests the function find_material_from_id() for cases with one id per
    material
    """
    mat_1 = F.Material(id=1, D_0=None, E_D=None)
    mat_2 = F.Material(id=2, D_0=None, E_D=None)
    my_Mats = F.Materials([mat_1, mat_2])
    assert my_Mats.find_material_from_id(1) == mat_1
    assert my_Mats.find_material_from_id(2) == mat_2


def test_find_material_from_id_with_several_ids():
    """Tests the function find_material_from_id() for cases with several ids
    per material
    """

    mat_1 = F.Material(id=[1, 2], D_0=None, E_D=None)
    my_Mats = F.Materials([mat_1])
    assert my_Mats.find_material_from_id(1) == mat_1
    assert my_Mats.find_material_from_id(2) == mat_1


def test_find_material_from_id_unfound_id():
    """
    Tests the function find_material_from_id with a list of materials
    without the searched ID
        - check that an error is rasied
    """
    mat_1 = F.Material(id=5, D_0=None, E_D=None)
    mat_2 = F.Material(id=2, D_0=None, E_D=None)
    mat_3 = F.Material(id=-1, D_0=None, E_D=None)

    my_Mats = F.Materials([mat_1, mat_2, mat_3])
    id_test = 1
    with pytest.raises(ValueError, match="Couldn't find ID {}".format(id_test)):
        my_Mats.find_material_from_id(id_test)


def test_find_material_from_name():
    """Checks the function find_material_from_name() returns the correct material"""
    mat_1 = F.Material(id=1, D_0=None, E_D=None, name="mat1")
    mat_2 = F.Material(id=2, D_0=None, E_D=None, name="mat2")
    my_Mats = F.Materials([mat_1, mat_2])
    assert my_Mats.find_material_from_name("mat1") == mat_1
    assert my_Mats.find_material_from_name("mat2") == mat_2


def test_find_material_from_name_unfound_name():
    """
    Check find_material_from_name raises an error when the name is not found
    """
    mat_1 = F.Material(id=5, D_0=None, E_D=None)
    mat_2 = F.Material(id=2, D_0=None, E_D=None)
    mat_3 = F.Material(id=-1, D_0=None, E_D=None)

    my_Mats = F.Materials([mat_1, mat_2, mat_3])
    name_test = "coucou"
    with pytest.raises(
        ValueError, match="No material with name {} was found".format(name_test)
    ):
        my_Mats.find_material_from_name(name_test)


def test_unused_thermal_cond():
    """
    Checks warnings when some keys are unused
    """

    mat_1 = F.Material(id=1, D_0=1, E_D=2, thermal_cond=2)
    my_mats = F.Materials([mat_1])
    with pytest.warns(UserWarning, match=r"thermal_cond key will be ignored"):
        my_mats.check_for_unused_properties(T=F.Temperature(100), derived_quantities=[])

    # this shouldn't throw warnings
    derived_quantities = [F.SurfaceFlux("T", surface=0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        my_mats.check_for_unused_properties(
            T=F.Temperature(100), derived_quantities=derived_quantities
        )


def test_missing_thermal_cond():
    """Tests that an error is raised when the thermal cond is missing"""
    my_mats = F.Materials([F.Material(1, D_0=1, E_D=1)])
    with pytest.raises(ValueError, match="Missing thermal_cond in materials"):
        my_mats.check_missing_properties(
            T=F.HeatTransferProblem(), derived_quantities=[]
        )


def test_missing_heat_capacity():
    """Tests that an error is raised when the heat_capacity is missing"""
    my_mats = F.Materials([F.Material(1, D_0=1, E_D=1, thermal_cond=1, rho=1)])
    with pytest.raises(ValueError, match="Missing heat_capacity in materials"):
        my_mats.check_missing_properties(
            T=F.HeatTransferProblem(), derived_quantities=[]
        )


def test_missing_rho():
    """Tests that an error is raised when the rho is missing"""
    my_mats = F.Materials(
        [F.Material(1, D_0=1, E_D=1, thermal_cond=1, heat_capacity=1)]
    )
    with pytest.raises(ValueError, match="Missing rho in materials"):
        my_mats.check_missing_properties(
            T=F.HeatTransferProblem(), derived_quantities=[]
        )


def test_different_ids_in_materials():
    """
    Checks that an error is raised when two materials have the same id
    """

    mat_1 = F.Material(id=1, D_0=1, E_D=2)
    mat_2 = F.Material(id=1, D_0=2, E_D=3)
    my_mats = F.Materials([mat_1, mat_2])
    with pytest.raises(ValueError, match=r"Some materials have the same id"):
        my_mats.check_unique_ids()


def test_unused_keys():
    """
    Checks warnings when some keys are unused
    """

    mat_1 = F.Material(id=1, D_0=1, E_D=2, rho=2)
    my_mats = F.Materials([mat_1])

    with pytest.warns(UserWarning, match=r"rho key will be ignored"):
        my_mats.check_for_unused_properties(T=F.Temperature(200), derived_quantities={})

    mat_1.rho = None
    mat_1.heat_capacity = 2

    with pytest.warns(UserWarning, match=r"heat_capacity key will be ignored"):
        my_mats.check_for_unused_properties(T=F.Temperature(200), derived_quantities={})


def test_non_matching_properties():
    mat_1 = F.Material(id=1, D_0=1, E_D=2, rho=2)
    mat_2 = F.Material(id=1, D_0=1, E_D=2)
    my_mats = F.Materials([mat_1, mat_2])
    with pytest.raises(ValueError, match=r"rho is not defined for all materials"):
        my_mats.check_consistency()


class TestCheckBorders:
    """General test for the check_borders method of the festim.Materials class"""

    def test_works(self):
        materials = [
            F.Material(id=1, D_0=None, E_D=None, borders=[0.5, 0.7]),
            F.Material(id=2, D_0=None, E_D=None, borders=[0, 0.5]),
        ]
        size = 0.7
        assert F.Materials(materials).check_borders(size) is True

    def test_not_beginning_at_zero(self):
        with pytest.raises(ValueError, match=r"zero"):
            size = 0.7
            materials = [
                F.Material(id=1, D_0=None, E_D=None, borders=[0.5, 0.7]),
                F.Material(id=1, D_0=None, E_D=None, borders=[0.2, 0.5]),
            ]
            F.Materials(materials).check_borders(size)

    def test_not_matching(self):
        with pytest.raises(ValueError, match=r"each other"):
            materials = [
                F.Material(id=1, D_0=None, E_D=None, borders=[0.5, 1]),
                F.Material(id=1, D_0=None, E_D=None, borders=[0, 0.6]),
                F.Material(id=1, D_0=None, E_D=None, borders=[0.6, 1]),
            ]
            size = 1
            F.Materials(materials).check_borders(size)

    def test_not_matching_with_size(self):
        with pytest.raises(ValueError, match=r"size"):
            materials = [
                F.Material(id=1, D_0=None, E_D=None, borders=[0, 1]),
            ]
            size = 3
            F.Materials(materials).check_borders(size)

    def test_1_material_2_subdomains(self):
        materials = F.Materials([F.Material([1, 2], 1, 0, borders=[[0, 1], [1, 9]])])

        materials.check_borders(size=9)

    def test_2_materials_3_subdomains(self):
        materials = F.Materials(
            [
                F.Material([1, 2], 1, 0, borders=[[0, 1], [1, 5]]),
                F.Material(3, 1, 0, borders=[5, 9]),
            ]
        )
        materials.check_borders(size=9)

    def test_1_material_1_id_2_borders(self):
        materials = F.Materials(
            [
                F.Material(1, 1, 0, borders=[[0, 1], [1, 9]]),
            ]
        )
        materials.check_borders(size=9)


def test_material_with_multiple_ids_solubility():
    """Tests the function find_material_from_id() for cases with several ids
    per material
    """

    mat_1 = F.Material(id=[1, 2], D_0=1, E_D=1)
    my_mats = F.Materials([mat_1])
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
    mat_1 = F.Material(
        1,
        D_0=1,
        E_D=0,
        S_0=7,
        E_S=0,
        thermal_cond=4,
        heat_capacity=5,
        rho=6,
        Q=11,
    )
    mat_2 = F.Material(
        2,
        D_0=2,
        E_D=0,
        S_0=8,
        E_S=0,
        thermal_cond=5,
        heat_capacity=6,
        rho=7,
        Q=12,
    )
    materials = F.Materials([mat_1, mat_2])
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
    Q = interpolate(materials.Q, DG_1)
    S = interpolate(materials.S, DG_1)

    for cell in cells(mesh):
        assert D(cell.midpoint().x()) == mf[cell]
        assert thermal_cond(cell.midpoint().x()) == mf[cell] + 3
        assert cp(cell.midpoint().x()) == mf[cell] + 4
        assert rho(cell.midpoint().x()) == mf[cell] + 5
        assert Q(cell.midpoint().x()) == mf[cell] + 10
        assert S(cell.midpoint().x()) == mf[cell] + 6


def test_E_S_without_S_0():
    with pytest.raises(ValueError, match="S_0 cannot be None"):
        F.Material(1, 1, 1, S_0=None, E_S=1)


def test_S_0_without_E_S():
    with pytest.raises(ValueError, match="E_S cannot be None"):
        F.Material(1, 1, 1, S_0=1, E_S=None)


def test_error_wrong_solubility_law_string():
    """Tests that an error is raised when the wrong value of solubility_law is set"""
    with pytest.raises(
        ValueError,
        match="Acceptable values for solubility_law are 'henry' and 'sievert'",
    ):
        F.Material(1, 1, 1, solubility_law="foo")


def test_equality_identity_two_empty_materials():
    """
    Tests equality and two of two empty F.Materials objects, i.e. checks
    that these F.Materials are equal but refer to different objects
    """
    my_materials1 = F.Materials([])
    my_materials2 = F.Materials([])
    assert (my_materials1 == my_materials2) and (my_materials1 is not my_materials2)


class TestMaterialsMethods:
    """Checks that F.Materials methods work properly"""

    my_mat1 = F.Material(1, 1, 0)
    my_mat2 = F.Material(2, 1, 0)

    my_materials = F.Materials([my_mat1])

    def test_mats_append(self):
        self.my_materials.append(self.my_mat2)
        assert self.my_materials == [self.my_mat1, self.my_mat2]

    def test_mats_insert(self):
        self.my_materials.insert(0, self.my_mat2)
        assert self.my_materials == [self.my_mat2, self.my_mat1, self.my_mat2]

    def test_mats_setitem(self):
        self.my_materials[0] = self.my_mat1
        assert self.my_materials == [self.my_mat1, self.my_mat1, self.my_mat2]

    def test_mats_extend_list_type(self):
        self.my_materials.extend([self.my_mat1])
        assert self.my_materials == [
            self.my_mat1,
            self.my_mat1,
            self.my_mat2,
            self.my_mat1,
        ]

    def test_mats_extend_self_type(self):
        self.my_materials.extend(F.Materials([self.my_mat2]))
        assert self.my_materials == F.Materials(
            [self.my_mat1, self.my_mat1, self.my_mat2, self.my_mat1, self.my_mat2]
        )


def test_set_materials_wrong_type():
    """Checks an error is raised when festim.Materials is set with the wrong type"""
    my_mat = F.Material(1, 1, 0)

    combinations = [my_mat, "coucou", 1, True]

    for mat_combination in combinations:
        with pytest.raises(
            TypeError,
            match="festim.Materials must be a list",
        ):
            F.Materials(mat_combination)

    with pytest.raises(
        TypeError,
        match="festim.Materials must be a list of festim.Material",
    ):
        F.Materials([my_mat, 2])


def test_assign_materials_wrong_type():
    """Checks an error is raised when the wrong type is assigned to festim.Materials"""
    my_mat = F.Material(1, 1, 0)
    my_materials = F.Materials([my_mat])

    combinations = ["coucou", 1, True]

    error_pattern = "festim.Materials must be a list of festim.Material"

    for mat_combination in combinations:
        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_materials.append(mat_combination)

        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_materials.extend([mat_combination])

        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_materials[0] = mat_combination

        with pytest.raises(
            TypeError,
            match=error_pattern,
        ):
            my_materials.insert(0, mat_combination)


class TestMaterialsPropertyDeprWarn:
    """
    A temporary test to check DeprecationWarnings in F.Materials.materials
    """

    my_mat = F.Material(id=1, E_D=1, D_0=1)
    my_mats = F.Materials([])

    def test_property_depr_warns(self):
        with pytest.deprecated_call():
            self.my_mats.materials

    def test_property_setter_depr_warns(self):
        with pytest.deprecated_call():
            self.my_mats.materials = [self.my_mat]


class TestMaterialsPropertyRaiseError:
    """
    A temporary test to check TypeErrors in F.Materials.materials
    """

    my_mat = F.Material(id=1, E_D=1, D_0=1)
    my_mats = F.Materials([])

    def test_set_materials_wrong_type(self):
        with pytest.raises(
            TypeError,
            match="materials must be a list",
        ):
            self.my_mats.materials = self.my_mat

    def test_set_materials_list_wrong_type(self):
        with pytest.raises(
            TypeError,
            match="materials must be a list of festim.Material",
        ):
            self.my_mats.materials = [self.my_mat, 1]


def test_instanciate_with_no_elements():
    """
    Test to catch bug described in issue #724
    """
    # define exports
    F.Materials()
