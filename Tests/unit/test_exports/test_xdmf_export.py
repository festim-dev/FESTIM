from FESTIM import XDMFExport
import fenics as f
import pytest


def test_define_file_upon_construction():
    mesh = f.UnitIntervalMesh(10)
    label = "my_label"
    folder = "my_folder"
    my_xdmf = XDMFExport("solute", label, folder)

    my_xdmf.file.write(mesh)

    mesh_in = f.Mesh()
    f.XDMFFile(folder + "/" + label + ".xdmf").read(mesh_in)


class TestDefineFile:
    mesh = f.UnitIntervalMesh(10)
    my_xdmf = XDMFExport("solute", "foo", "foo")

    def test_file_exists(self):
        label = "my_label"
        folder = "my_folder"
        self.my_xdmf.label = label
        self.my_xdmf.folder = folder
        self.my_xdmf.define_xdmf_file()
        self.my_xdmf.file.write(self.mesh)

        mesh_in = f.Mesh()
        f.XDMFFile(folder + "/" + label + ".xdmf").read(mesh_in)


class TestWrite:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    u = f.Function(V)
    label_to_function = {"solute": u}

    def test_no_checkpoint_no_error(self):
        my_xdmf = XDMFExport("solute", "foo", "foo")
        my_xdmf.write(self.label_to_function, t=2)

    def test_no_checkpoint_error_on_read(self):
        """checks that without checkpointing one cannot read the
        xdmf file
        """
        my_xdmf = XDMFExport("solute", "foo", "foo", checkpoint=False)
        my_xdmf.write(self.label_to_function, t=2)
        u2 = f.Function(self.V)
        with pytest.raises(TypeError, match="incompatible function arguments"):
            my_xdmf.file.read(u2)

    def test_checkpointing(self):
        """checks that the xdmf file can be read with checkpointing
        """
        my_xdmf = XDMFExport("solute", "foo", "foo")
        my_xdmf.write(self.label_to_function, t=2)
        u2 = f.Function(self.V)
        my_xdmf.file.read_checkpoint(u2, "foo", 0)
