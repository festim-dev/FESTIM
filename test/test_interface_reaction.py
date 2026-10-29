import pytest

import festim as F

mat = F.Material(D_0=1, E_D=0)
vol_0 = F.VolumeSubdomain1D(id=1, material=mat, borders=[0, 0.5])
vol_1 = F.VolumeSubdomain1D(id=2, material=mat, borders=[0.5, 1])


@pytest.mark.parametrize(
    "reactant_subdomains, product_subdomains, match",
    [
        ([vol_1], [vol_1], "reactant A .* must be defined in subdomain 1"),
        (None, [vol_1], "reactant A .* must be defined in subdomain 1"),
        ([vol_0], [vol_0], "product B .* must be defined in subdomain 2"),
        ([vol_0], None, "product B .* must be defined in subdomain 2"),
    ],
)
def test_get_formulation_raises_when_species_on_wrong_side(
    reactant_subdomains, product_subdomains, match
):
    """Test that a reactant missing from the first subdomain, or a product missing
    from the second, raises a ValueError naming the species and the subdomain."""
    # BUILD
    A = F.Species("A", subdomains=reactant_subdomains)
    B = F.Species("B", subdomains=product_subdomains)
    interface = F.InterfaceReaction(
        id=3,
        subdomains=[vol_0, vol_1],
        k_plus=1,
        k_minus=1,
        reactants=[A],
        products=[B],
    )

    # RUN & TEST
    with pytest.raises(ValueError, match=match):
        interface.get_formulation(dS=None, species=[], temperature=None)


A = F.Species("A", subdomains=[vol_0])
B = F.Species("B", subdomains=[vol_1])


def make_reaction(reactants=(A,), products=(B,)):
    """An InterfaceReaction between vol_0 and vol_1 with unit rate coefficients."""
    reactants = list(reactants) if isinstance(reactants, tuple) else reactants
    products = list(products) if isinstance(products, tuple) else products
    return F.InterfaceReaction(
        id=3,
        subdomains=[vol_0, vol_1],
        k_plus=1,
        k_minus=1,
        reactants=reactants,
        products=products,
    )


def test_single_reactant_is_stored_as_list():
    """Test that a single Species given as reactants is stored as a list."""
    assert make_reaction(reactants=A).reactants == [A]


def test_single_product_is_stored_as_list():
    """Test that a single Species given as products is stored as a list."""
    assert make_reaction(products=B).products == [B]


@pytest.mark.parametrize("products", [None, []])
def test_no_products_is_stored_as_empty_list(products):
    """Test that None or an empty list of products gives an irreversible reaction
    with an empty list of products."""
    assert make_reaction(products=products).products == []


def test_empty_reactants_raise():
    """Test that an empty list of reactants raises a ValueError."""
    with pytest.raises(ValueError, match="reactants must contain at least one"):
        make_reaction(reactants=[])


@pytest.mark.parametrize(
    "reactants",
    [1, "A", None, [A, 1], [F.ImplicitSpecies(n=1, name="empty")]],
)
def test_reactants_of_wrong_type_raise(reactants):
    """Test that reactants other than Species raise a TypeError."""
    with pytest.raises(TypeError, match=r"reactants must be a festim\.Species"):
        make_reaction(reactants=reactants)


@pytest.mark.parametrize("products", [1, "B", [B, 1], [B, None]])
def test_products_of_wrong_type_raise(products):
    """Test that products other than Species raise a TypeError."""
    with pytest.raises(TypeError, match=r"products must be a festim\.Species"):
        make_reaction(products=products)
