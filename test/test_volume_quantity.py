import pytest

import festim as F

dummy_mat = F.Material(D_0=1, E_D=1, name="dummy")
dummy_volume = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=dummy_mat)


def test_filename_setter_raises_TypeError():
    """Test that a TypeError is raised when the filename is not a string"""

    with pytest.raises(TypeError, match="filename must be of type str"):
        F.VolumeQuantity(
            filename=1,
            field=F.Species("test"),
            volume=dummy_volume,
        )


def test_filename_setter_raises_ValueError():
    """Test that a ValueError is raised when the filename does not end with .csv or .txt"""

    with pytest.raises(ValueError):
        F.VolumeQuantity(
            filename="my_export.xdmf",
            field=F.Species("test"),
            volume=dummy_volume,
        )


def test_field_setter_raises_TypeError():
    """Test that a TypeError is raised when the field is not a F.Species"""

    with pytest.raises(TypeError):
        F.VolumeQuantity(
            field=1,
            volume=dummy_volume,
        )
