from FESTIM import TXTExport
import fenics as f
import os


class TestWrite:
    my_export = TXTExport("solute", [1, 2, 3], "solute_label", "my_folder")
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    u = f.Function(V)

    label_to_function = {"solute": u}

    def test_file_exists(self):
        current_time = 1
        self.my_export.write(self.label_to_function, current_time=current_time, dt=f.Constant(3))

        assert os.path.exists("{}/{}_{}.txt".format(self.my_export.folder, self.my_export.label, current_time))

    def test_file_doesnt_exist(self):
        current_time = 10
        self.my_export.write(self.label_to_function, current_time=current_time, dt=f.Constant(3))

        assert not os.path.exists("{}/{}_{}.txt".format(self.my_export.folder, self.my_export.label, current_time))

    def test_dt_is_changed(self):
        current_time = 1
        initial_value = 10
        dt = f.Constant(initial_value)
        self.my_export.write(self.label_to_function, current_time=current_time, dt=dt)

        assert float(dt) == self.my_export.when_is_next_time(current_time) - current_time


class TestIsItTimeToExport:
    my_export = TXTExport("solute", [1, 2, 3], "solute_label", "my_folder")

    def test_true(self):
        assert self.my_export.is_it_time_to_export(2)
        assert self.my_export.is_it_time_to_export(3)
        assert self.my_export.is_it_time_to_export(1)

    def test_false(self):
        assert not self.my_export.is_it_time_to_export(-1)
        assert not self.my_export.is_it_time_to_export(0)
        assert not self.my_export.is_it_time_to_export(5)
        assert not self.my_export.is_it_time_to_export(1.5)


class TestWhenIsNextTime:
    my_export = TXTExport("solute", [1, 2, 3], "solute_label", "my_folder")

    def test_there_is_a_next_time(self):
        assert self.my_export.when_is_next_time(2) == 3
        assert self.my_export.when_is_next_time(1) == 2
        assert self.my_export.when_is_next_time(0) == 1
        assert self.my_export.when_is_next_time(0.5) == 1

    def test_last(self):
        assert self.my_export.when_is_next_time(3) is None
        assert self.my_export.when_is_next_time(4) is None
