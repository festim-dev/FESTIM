import pytest

from festim.material import SolubilityLaw
from festim.subdomain.interface import interface_condition_term


@pytest.mark.parametrize(
    "law",
    [SolubilityLaw.SIEVERT, SolubilityLaw.HENRY, SolubilityLaw.NONE],
)
def test_interface_condition_term_same_law_returns_ratio(law):
    """Identical solubility laws on both sides give the linear term c / K_S."""
    assert interface_condition_term(6.0, 3.0, law, law) == 2.0


def test_interface_condition_term_henry_side_of_mixed_interface_returns_ratio():
    """A Henry side facing a different law gives the linear term c / K_S."""
    result = interface_condition_term(
        6.0, 3.0, SolubilityLaw.HENRY, SolubilityLaw.SIEVERT
    )
    assert result == 2.0


def test_interface_condition_term_sievert_side_of_mixed_interface_returns_squared():
    """A Sievert side facing a different law gives the squared term (c / K_S) ** 2."""
    result = interface_condition_term(
        6.0, 3.0, SolubilityLaw.SIEVERT, SolubilityLaw.HENRY
    )
    assert result == 4.0


def test_interface_condition_term_raises_for_unsupported_mixed_law():
    """An unsupported law (NONE) facing a different law raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported solubility law"):
        interface_condition_term(6.0, 3.0, SolubilityLaw.NONE, SolubilityLaw.HENRY)
