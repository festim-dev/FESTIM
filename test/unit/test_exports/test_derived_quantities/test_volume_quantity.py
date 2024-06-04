import festim as F
import pytest


@pytest.mark.parametrize("volume", [2.1, [0], [1, 1], "coucou", ["coucou"], True])
def test_wrong_type_for_volume(volume):
    """
    Tests that error is raised when the surface volume is set with wrong type

    Args:
        volume (): wrong type for the surface attribute
    """
    with pytest.raises(TypeError, match="volume should be an int"):
        F.VolumeQuantity(field=0, volume=volume)
