from festim import TXTExport, Material
import fenics as f
import os
import pytest
import numpy as np
from pathlib import Path


class TestWrite:
    @pytest.fixture
    def mesh(self):
        mesh = f.UnitIntervalMesh(10)

        return mesh

    @pytest.fixture
    def function(self):
        mesh = f.UnitIntervalMesh(10)
        V = f.FunctionSpace(mesh, "P", 1)
        u = f.Function(V)

        return u

    @pytest.fixture
    def function_subspace(self):
        mesh = f.UnitIntervalMesh(10)
        V = f.VectorFunctionSpace(mesh, "P", 1, 2)
        u = f.Function(V)

        return u.sub(0)

    @pytest.fixture
    def my_export(self, tmpdir):
        d = tmpdir.mkdir("test_folder")
        my_export = TXTExport(
            "solute",
            times=[1, 2, 3],
            filename="{}/solute_label.txt".format(str(Path(d))),
        )

        return my_export

    def test_file_exists(self, my_export, function, mesh):
        current_time = 1
        my_export.function = function
        my_export.initialise(mesh)
        my_export.write(
            current_time=current_time,
            final_time=None,
        )

        assert os.path.exists(my_export.filename)

    def test_file_doesnt_exist(self, my_export, function, mesh):
        current_time = 10
        my_export.function = function
        my_export.initialise(mesh)
        my_export.write(
            current_time=current_time,
            final_time=None,
        )

        assert not os.path.exists(my_export.filename)

    def test_create_folder(self, my_export, function, mesh):
        """Checks that write() creates the folder if it doesn't exist"""
        current_time = 1
        my_export.function = function
        slash_indx = my_export.filename.rfind("/")
        my_export.filename = (
            my_export.filename[:slash_indx]
            + "/folder2"
            + my_export.filename[slash_indx:]
        )
        my_export.initialise(mesh)
        my_export.write(
            current_time=current_time,
            final_time=None,
        )

        assert os.path.exists(my_export.filename)

    def test_subspace(self, my_export, function_subspace, mesh):
        current_time = 1
        my_export.function = function_subspace
        my_export.initialise(mesh)
        my_export.write(
            current_time=current_time,
            final_time=None,
        )

        assert os.path.exists(my_export.filename)

    def test_error_filename_endswith_txt(self, my_export):
        with pytest.raises(ValueError, match="filename must end with .txt"):
            my_export.filename = "coucou"

    def test_error_filename_not_a_str(self, my_export):
        with pytest.raises(TypeError, match="filename must be a string"):
            my_export.filename = 2

    def test_sorted_by_x(self, my_export, function, mesh):
        """Checks that the exported data is sorted by x"""
        current_time = 1
        my_export.function = function
        my_export.initialise(mesh)
        my_export.write(
            current_time=current_time,
            final_time=None,
        )
        assert (np.diff(my_export.data[:, 0]) >= 0).all()

    @pytest.mark.parametrize(
        "materials,project_to_DG,filter,export_len",
        [
            (None, False, False, 11),
            (
                [
                    Material(id=1, D_0=1, E_D=0, S_0=1, E_S=0, borders=[0, 0.5]),
                    Material(id=2, D_0=2, E_D=0, S_0=2, E_S=0, borders=[0.5, 1]),
                ],
                True,
                True,
                12,  # + 1 duplicate near the interface
            ),
            (
                [
                    Material(id=1, D_0=1, E_D=0, S_0=1, E_S=0, borders=[0, 0.5]),
                    Material(id=2, D_0=2, E_D=0, S_0=2, E_S=0, borders=[0.5, 1]),
                ],
                True,
                False,
                20,  # 2 * (len_vertices - 1)
            ),
        ],
    )
    def test_duplicates(
        self, materials, project_to_DG, filter, export_len, my_export, function, mesh
    ):
        """
        Checks that the exported data does not contain duplicates
        except those near interfaces
        """
        current_time = 1
        my_export.function = function
        my_export.filter = filter
        my_export.initialise(mesh, project_to_DG, materials)
        my_export.write(
            current_time=current_time,
            final_time=None,
        )

        assert len(my_export.data) == export_len


class TestIsItTimeToExport:
    @pytest.fixture
    def my_export(self, tmpdir):
        d = tmpdir.mkdir("test_folder")
        my_export = TXTExport(
            "solute",
            times=[1, 2, 3],
            filename="{}/solute_label.txt".format(str(Path(d))),
        )

        return my_export

    def test_true(self, my_export):
        assert my_export.is_it_time_to_export(2)
        assert my_export.is_it_time_to_export(3)
        assert my_export.is_it_time_to_export(1)

    def test_false(self, my_export):
        assert not my_export.is_it_time_to_export(-1)
        assert not my_export.is_it_time_to_export(0)
        assert not my_export.is_it_time_to_export(5)
        assert not my_export.is_it_time_to_export(1.5)


class TestIsLast:
    @pytest.fixture
    def my_export(self, tmpdir):
        d = tmpdir.mkdir("test_folder")
        my_export = TXTExport(
            "solute",
            filename="{}/solute_label.txt".format(str(Path(d))),
        )

        return my_export

    def test_final_time_is_none(self, my_export):
        assert my_export.is_last(1, None) == True

    @pytest.mark.parametrize("current_time,output", [(1, False), (3, False), (5, True)])
    def test_times_is_none(self, current_time, output, my_export):
        final_time = 5
        assert my_export.is_last(current_time, final_time) == output

    @pytest.mark.parametrize("current_time,output", [(1, False), (2, False), (3, True)])
    def test_times_not_none(self, current_time, output, my_export):
        my_export.times = [1, 2, 3]
        final_time = 5
        assert my_export.is_last(current_time, final_time) == output
