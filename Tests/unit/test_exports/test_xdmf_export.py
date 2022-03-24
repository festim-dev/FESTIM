from FESTIM import XDMFExport
import fenics as f
import pytest
from pathlib import Path


@pytest.fixture
def folder(tmpdir):
    return str(Path(tmpdir.mkdir("test_folder")))


def test_define_file_upon_construction(folder):
    mesh = f.UnitIntervalMesh(10)
    label = "my_label"
    my_xdmf = XDMFExport("solute", label, folder)

    my_xdmf.file.write(mesh)

    mesh_in = f.Mesh()
    f.XDMFFile(folder + "/" + label + ".xdmf").read(mesh_in)


class TestDefineFile:
    mesh = f.UnitIntervalMesh(10)
    my_xdmf = XDMFExport("solute", "foo", "foo")

    def test_file_exists(self, folder):
        label = "my_label"
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

    def test_no_checkpoint_no_error(self, folder):
        my_xdmf = XDMFExport("solute", "foo", folder)
        my_xdmf.function = self.u

        my_xdmf.write(t=2)

    def test_no_checkpoint_error_on_read(self, folder):
        """checks that without checkpointing one cannot read the
        xdmf file
        """
        my_xdmf = XDMFExport("solute", "foo", folder, checkpoint=False)
        my_xdmf.function = self.u

        my_xdmf.write(t=2)
        u2 = f.Function(self.V)
        with pytest.raises(TypeError, match="incompatible function arguments"):
            my_xdmf.file.read(u2)

    def test_checkpointing(self, folder):
        """checks that the xdmf file can be read with checkpointing
        """
        my_xdmf = XDMFExport("solute", "foo", folder)
        my_xdmf.function = self.u

        my_xdmf.write(t=2)
        u2 = f.Function(self.V)
        my_xdmf.file.read_checkpoint(u2, "foo", 0)


def test_error_folder_empty_str():
    with pytest.raises(ValueError, match="empty string"):
        XDMFExport("solute", "solute", "")


def test_error_folder_not_a_str():
    with pytest.raises(TypeError, match="type str"):
        XDMFExport("solute", "solute", 2)


def test_error_checkpoint_wrong_type():
    with pytest.raises(TypeError, match="checkpoint should be a bool"):
        XDMFExport("solute", "solute", "my_folder", checkpoint=2)


def test_wrong_argument_for_mode():
    accepted_values_msg = "accepted values for mode are int and 'last'"
    # test wrong type
    with pytest.raises(ValueError, match=accepted_values_msg):
        XDMFExport("solute", "mobile", "out", mode=1.2)
    with pytest.raises(ValueError, match=accepted_values_msg):
        XDMFExport("solute", "mobile", "out", mode=[1, 2, 3])

    # test wrong string
    with pytest.raises(ValueError, match=accepted_values_msg):
        XDMFExport("solute", "mobile", "out", mode="foo")

    # test negative integer
    with pytest.raises(ValueError, match="mode must be positive"):
        XDMFExport("solute", "mobile", "out", mode=-1)

    # test mode=0
    with pytest.raises(ValueError, match="mode must be positive"):
        XDMFExport("solute", "mobile", "out", mode=0)
