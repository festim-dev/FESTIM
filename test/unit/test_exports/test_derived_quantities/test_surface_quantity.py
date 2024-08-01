import festim as F
import pytest


@pytest.mark.parametrize("surface", [2.1, [0], [1, 1], "coucou", ["coucou"], True])
def test_wrong_type_for_surface(surface):
    """
    Tests that error is raised when the surface attribute is set with wrong type

    Args:
        surface (): wrong type for the surface attribute
    """
    with pytest.raises(TypeError, match="surface should be an int"):
        F.SurfaceQuantity(field=0, surface=surface)
