from FESTIM import XDMFExport, extract_xdmf_labels
import fenics as f
import pytest
from pathlib import Path


@pytest.fixture
def folder(tmpdir):
    return str(Path(tmpdir.mkdir("test_folder")))


def test_define_file_upon_construction(folder):
    mesh = f.UnitIntervalMesh(10)
    filename = "{}/my_filename.xdmf".format(folder)
    my_xdmf = XDMFExport("solute", label="solute_label", filename=filename)

    my_xdmf.file.write(mesh)

    mesh_in = f.Mesh()
    f.XDMFFile(filename).read(mesh_in)


class TestDefineFile:
    mesh = f.UnitIntervalMesh(10)
    my_xdmf = XDMFExport("solute", "my_label", "my_filename.xdmf")

    def test_file_exists(self, folder):
        filename = "{}/my_filename.xdmf".format(folder)
        self.my_xdmf.filename = filename
        self.my_xdmf.define_xdmf_file()
        self.my_xdmf.file.write(self.mesh)

        mesh_in = f.Mesh()
        f.XDMFFile(filename).read(mesh_in)


class TestWrite:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)
    u = f.Function(V)
    label_to_function = {"solute": u}

    def test_no_checkpoint_no_error(self, folder):
        filename = folder + "/coucou.xdmf"
        my_xdmf = XDMFExport("solute", "foo", filename)
        my_xdmf.function = self.u

        my_xdmf.write(t=2)

    def test_no_checkpoint_error_on_read(self, folder):
        """checks that without checkpointing one cannot read the
        xdmf file
        """
        filename = folder + "/coucou.xdmf"
        my_xdmf = XDMFExport("solute", "foo", filename, checkpoint=False)
        my_xdmf.function = self.u

        my_xdmf.write(t=2)
        u2 = f.Function(self.V)
        with pytest.raises(TypeError, match="incompatible function arguments"):
            my_xdmf.file.read(u2)

    def test_checkpointing(self, folder):
        """checks that the xdmf file can be read with checkpointing"""
        filename = folder + "/coucou.xdmf"
        my_xdmf = XDMFExport("solute", "foo", filename)
        my_xdmf.function = self.u

        my_xdmf.write(t=2)
        u2 = f.Function(self.V)
        my_xdmf.file.read_checkpoint(u2, "foo", 0)

    def test_write_attribute(self, folder):
        """Checks that the file is written with the appropriate attribute"""
        filename = folder + "/coucou.xdmf"
        my_xdmf = XDMFExport("solute", "coucou", filename)
        my_xdmf.function = self.u
        my_xdmf.write(t=0)

        labels = extract_xdmf_labels(filename)

        assert len(labels) == 1
        assert labels[0] == "coucou"


def test_error_folder_endswith_xdmf():
    with pytest.raises(ValueError, match="must end with .xdmf"):
        XDMFExport("solute", "solute", "coucou")


def test_error_folder_not_a_str():
    with pytest.raises(TypeError, match="must be a string"):
        XDMFExport("solute", "solute", 2)


def test_error_checkpoint_wrong_type():
    with pytest.raises(TypeError, match="checkpoint should be a bool"):
        XDMFExport("solute", "solute", "my_filename.xdmf", checkpoint=2)


def test_wrong_argument_for_mode():
    accepted_values_msg = "accepted values for mode are int and 'last'"
    # test wrong type
    with pytest.raises(ValueError, match=accepted_values_msg):
        XDMFExport("solute", "mobile", "out.xdmf", mode=1.2)
    with pytest.raises(ValueError, match=accepted_values_msg):
        XDMFExport("solute", "mobile", "out.xdmf", mode=[1, 2, 3])

    # test wrong string
    with pytest.raises(ValueError, match=accepted_values_msg):
        XDMFExport("solute", "mobile", "out.xdmf", mode="foo")

    # test negative integer
    with pytest.raises(ValueError, match="mode must be positive"):
        XDMFExport("solute", "mobile", "out.xdmf", mode=-1)

    # test mode=0
    with pytest.raises(ValueError, match="mode must be positive"):
        XDMFExport("solute", "mobile", "out.xdmf", mode=0)
