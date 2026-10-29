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
