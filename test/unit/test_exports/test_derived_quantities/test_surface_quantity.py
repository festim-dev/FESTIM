from nis import match
import festim as F
import pytest


def test_wrong_type_for_surface():
    for surface in [2.1, [0], [1, 1], "coucou", ["coucou"], True]:
        with pytest.raises(TypeError, match="surface should be an int"):
            print(surface)
            F.SurfaceQuantity(field=0, surface=surface)
