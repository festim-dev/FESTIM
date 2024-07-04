from festim import AdsorbedHydrogen
from .tools import c_0D, mesh_1D
import fenics as f


def test_field_is_adsorbed():
    """
    Tests that the festim.SurfaceQuantity field is set to adsorbed
    when festim.AdsorbedHydrogen is used
    """

    my_quantity = AdsorbedHydrogen(1)
    assert my_quantity.field == "adsorbed"


def test_title_with_units():
    my_quantity = AdsorbedHydrogen(1)
    my_quantity.function = c_0D
    my_quantity.show_units = True
    assert my_quantity.title == "Adsorbed H on surface 1 (H m-2)"


def test_title_without_units():
    my_quantity = AdsorbedHydrogen(1)
    assert my_quantity.title == "Adsorbed H on surface 1"


def test_compute():
    """Test that the adsorbed hydrogen export computes the correct value"""

    surface_markers = f.MeshFunction("size_t", mesh_1D, 0)
    left = f.CompiledSubDomain("near(x[0], 0) && on_boundary")
    left.mark(surface_markers, 1)

    my_quantity = AdsorbedHydrogen(1)
    ds = f.Measure("ds", domain=mesh_1D, subdomain_data=surface_markers)
    my_quantity.ds = ds
    my_quantity.function = c_0D

    produced = my_quantity.compute()
    assert produced == f.assemble(c_0D * ds(1))
