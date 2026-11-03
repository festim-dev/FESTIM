import warnings

import pytest

import festim as F

mat = F.Material(D_0=1, E_D=0)
vol_0 = F.VolumeSubdomain1D(id=1, material=mat, borders=[0, 0.5])
vol_1 = F.VolumeSubdomain1D(id=2, material=mat, borders=[0.5, 1])
ONE_SIDED = "lives on volume subdomain"


def species_of(model, interface):
    """Species coupled by ``interface``, turning any warning into an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return model.interface_species(interface)


def make_model(species, interfaces):
    return F.HydrogenTransportProblemDiscontinuous(
        species=species, interfaces=interfaces
    )


def test_species_on_both_sides_is_coupled():
    """Test that a species defined on both subdomains is coupled by an Interface."""
    H = F.Species("H", subdomains=[vol_0, vol_1])
    interface = F.Interface(id=3, subdomains=[vol_0, vol_1])
    assert species_of(make_model([H], [interface]), interface) == [H]


def test_one_sided_species_warns():
    """Test that a species on one side only, with no reaction on the interface,
    raises a warning."""
    A = F.Species("A", subdomains=[vol_0])
    interface = F.Interface(id=3, subdomains=[vol_0, vol_1])
    with pytest.warns(UserWarning, match=ONE_SIDED):
        make_model([A], [interface]).interface_species(interface)


@pytest.mark.filterwarnings(f"ignore:.*{ONE_SIDED}")
def test_one_sided_species_is_not_coupled():
    """Test that a species on one side only is not coupled by an Interface."""
    A = F.Species("A", subdomains=[vol_0])
    interface = F.Interface(id=3, subdomains=[vol_0, vol_1])
    assert make_model([A], [interface]).interface_species(interface) == []


@pytest.mark.parametrize("side", ["reactant", "product"])
def test_species_of_a_reaction_on_the_same_id_does_not_warn(side):
    """Test that a one-sided species taking part in an InterfaceReaction on the
    same facets raises no warning."""
    A = F.Species("A", subdomains=[vol_0])
    B = F.Species("B", subdomains=[vol_1])
    interface = F.Interface(id=3, subdomains=[vol_0, vol_1])
    reaction = F.InterfaceReaction(
        id=3, subdomains=[vol_0, vol_1], k_plus=1, k_minus=1, reactants=A, products=B
    )
    model = make_model([A] if side == "reactant" else [B], [interface, reaction])
    assert species_of(model, interface) == []


def test_species_of_a_reaction_on_another_id_still_warns():
    """Test that a one-sided species only reacting on other facets still raises a
    warning for this interface."""
    A = F.Species("A", subdomains=[vol_0])
    interface = F.Interface(id=3, subdomains=[vol_0, vol_1])
    reaction = F.InterfaceReaction(
        id=4, subdomains=[vol_0, vol_1], k_plus=1, k_minus=1, reactants=A, products=None
    )
    with pytest.warns(UserWarning, match=ONE_SIDED):
        make_model([A], [interface, reaction]).interface_species(interface)
