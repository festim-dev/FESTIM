from FESTIM import TXTExports
import fenics as f
import os


class TestWrite:
    my_export = TXTExports(["solute", "T"], [1, 2, 3], ["solute_label", "T_label"], "my_folder")
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    u = f.Function(V)
    T = f.Function(V)

    label_to_function = {"solute": u, "T": T}

    def test_txt_exports_times(self):
        for export in self.my_export.exports:
            assert export.times == self.my_export.times

    def test_file_exists(self):
        current_time = 1
        self.my_export.write(self.label_to_function, current_time=current_time, dt=f.Constant(3))
        for export in self.my_export.exports:
            assert os.path.exists("{}/{}_{}.txt".format(self.my_export.folder, export.label, current_time))

    def test_file_doesnt_exist(self):
        current_time = 10
        self.my_export.write(self.label_to_function, current_time=current_time, dt=f.Constant(3))
        for export in self.my_export.exports:
            assert not os.path.exists("{}/{}_{}.txt".format(self.my_export.folder, export.label, current_time))

    def test_dt_is_changed(self):
        current_time = 1
        initial_value = 10
        dt = f.Constant(initial_value)
        self.my_export.write(self.label_to_function, current_time=current_time, dt=dt)

        assert float(dt) == 2 - current_time
