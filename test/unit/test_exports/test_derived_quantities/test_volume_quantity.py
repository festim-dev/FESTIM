from nis import match
import festim as F
import pytest


def test_wrong_type_for_volume():
    for volume in [2.1, [0], [1, 1], "coucou", ["coucou"], True]:
        with pytest.raises(TypeError, match="volume should be an int"):
            print(volume)
            F.VolumeQuantity(field=0, volume=volume)
