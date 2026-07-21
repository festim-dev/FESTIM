import pytest

import festim as F

dummy_mat = F.Material(D_0=1, E_D=1, name="dummy")
dummy_volume = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=dummy_mat)


def test_field_setter_raises_TypeError():
    """Test that a TypeError is raised when the field is not a F.Species."""

    with pytest.raises(TypeError):
        F.VolumeQuantity(
            field=1,
            volume=dummy_volume,
        )
