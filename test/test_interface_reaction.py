from types import SimpleNamespace

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F
from festim.subdomain.interface import compute_ordered_interior_facet_data

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


def make_reaction(reactants=(A,), products=(B,), k_plus=1, k_minus=1):
    """An InterfaceReaction between vol_0 and vol_1, with unit rates by default."""
    reactants = list(reactants) if isinstance(reactants, tuple) else reactants
    products = list(products) if isinstance(products, tuple) else products
    return F.InterfaceReaction(
        id=3,
        subdomains=[vol_0, vol_1],
        k_plus=k_plus,
        k_minus=k_minus,
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


def test_products_are_optional():
    """Test that an InterfaceReaction built without products has none."""
    reaction = F.InterfaceReaction(
        id=3, subdomains=[vol_0, vol_1], k_plus=1, k_minus=0, reactants=[A]
    )
    assert reaction.products == []


@pytest.mark.parametrize("value", [0, 1, 2.5, np.float64(3.0)])
@pytest.mark.parametrize("rate", ["k_plus", "k_minus"])
def test_valid_rate_is_stored(rate, value):
    """Test that a non-negative number is accepted as a rate coefficient."""
    assert getattr(make_reaction(**{rate: value}), rate) == value


@pytest.mark.parametrize("value", ["1", None, [1], lambda T: T])
@pytest.mark.parametrize("rate", ["k_plus", "k_minus"])
def test_rate_of_wrong_type_raises(rate, value):
    """Test that a rate coefficient that is not a number raises a TypeError."""
    with pytest.raises(TypeError, match=f"{rate} must be a float or an int"):
        make_reaction(**{rate: value})


@pytest.mark.parametrize("rate", ["k_plus", "k_minus"])
def test_negative_rate_raises(rate):
    """Test that a negative rate coefficient raises a ValueError."""
    with pytest.raises(ValueError, match=f"{rate} must be non-negative"):
        make_reaction(**{rate: -1})


@pytest.mark.parametrize("rate", ["k_plus", "k_minus"])
def test_rate_is_validated_on_assignment(rate):
    """Test that setting a rate coefficient after construction is validated too."""
    reaction = make_reaction()
    with pytest.raises(ValueError, match=f"{rate} must be non-negative"):
        setattr(reaction, rate, -1)


DECOY = 1e3  # value a field takes on the wrong side of the interface
K_PLUS, K_MINUS = 11, 13


@pytest.fixture
def two_cells():
    """Two cells meeting at x = 0.5, vol_0 on the left and vol_1 on the right: a DG0
    space, the cell of each subdomain, and a dS measure on the shared facet (id 3)
    ordered with vol_0 on "+"."""
    mesh = dolfinx.mesh.create_interval(MPI.COMM_SELF, 2, [0, 1])
    cell_0 = dolfinx.mesh.locate_entities(mesh, 1, lambda x: x[0] <= 0.5)[0]
    cell_1 = dolfinx.mesh.locate_entities(mesh, 1, lambda x: x[0] >= 0.5)[0]
    order = np.argsort([cell_0, cell_1])
    cells = np.array([cell_0, cell_1], dtype=np.int32)[order]
    tags = np.array([vol_0.id, vol_1.id], dtype=np.int32)[order]
    cell_tags = dolfinx.mesh.meshtags(mesh, 1, cells, tags)
    facet = dolfinx.mesh.locate_entities(mesh, 0, lambda x: np.isclose(x[0], 0.5))
    facet_tags = dolfinx.mesh.meshtags(mesh, 0, facet, np.array([3], dtype=np.int32))
    data = compute_ordered_interior_facet_data(cell_tags, facet_tags, 3, vol_0, vol_1)
    V = dolfinx.fem.functionspace(mesh, ("DG", 0))
    return SimpleNamespace(
        V=V,
        dof={
            vol_0: V.dofmap.cell_dofs(cell_0)[0],
            vol_1: V.dofmap.cell_dofs(cell_1)[0],
        },
        dS=ufl.Measure("dS", domain=mesh, subdomain_data=[data]),
    )


def make_species(two_cells):
    """Species A = 2 and C = 3 on vol_0, B = 5 and D = 7 on vol_1, each a DG0 field
    equal to DECOY on the other side, with a DG0 test function."""
    species = {}
    for name, volume, value in [
        ("A", vol_0, 2),
        ("C", vol_0, 3),
        ("B", vol_1, 5),
        ("D", vol_1, 7),
    ]:
        u = dolfinx.fem.Function(two_cells.V)
        u.x.array[:] = DECOY
        u.x.array[two_cells.dof[volume]] = value
        spe = F.Species(name, subdomains=[volume])
        spe.subdomain_to_solution[volume] = u
        spe.subdomain_to_test_function[volume] = ufl.TestFunction(two_cells.V)
        species[name] = spe
    return species


def make_wired_reaction(two_cells, reactants, products, subdomains=(vol_0, vol_1)):
    """An InterfaceReaction with K_PLUS, K_MINUS between the named species."""
    species = make_species(two_cells)
    return F.InterfaceReaction(
        id=3,
        subdomains=list(subdomains),
        k_plus=K_PLUS,
        k_minus=K_MINUS,
        reactants=[species[name] for name in reactants],
        products=[species[name] for name in products],
    )


def rate_on_interface(reaction, two_cells):
    """The reaction rate integrated over the interface point (of measure 1)."""
    form = dolfinx.fem.form(reaction.reaction_term() * two_cells.dS(3))
    return dolfinx.fem.assemble_scalar(form)


def residual_on(form, volume, two_cells):
    """Entry of the assembled residual on the cell of ``volume``."""
    vector = dolfinx.fem.assemble_vector(dolfinx.fem.form(form))
    return vector.array[two_cells.dof[volume]]


@pytest.mark.parametrize(
    "reactants, products, expected",
    [
        (["A"], ["B"], K_PLUS * 2 - K_MINUS * 5),
        (["A", "A"], ["B"], K_PLUS * 2**2 - K_MINUS * 5),
        (["A"], ["B", "B"], K_PLUS * 2 - K_MINUS * 5**2),
        (["A", "C"], ["B", "D"], K_PLUS * 2 * 3 - K_MINUS * 5 * 7),
        (["A"], [], K_PLUS * 2),
    ],
)
def test_reaction_term_follows_mass_action(two_cells, reactants, products, expected):
    """Test that the rate is k_plus times the product of the reactants on vol_0
    minus k_minus times the product of the products on vol_1, reading each field
    on its own side of the interface."""
    reaction = make_wired_reaction(two_cells, reactants, products)
    assert np.isclose(rate_on_interface(reaction, two_cells), expected)


def test_reaction_term_follows_swapped_restriction(two_cells):
    """Test that a reaction declared from vol_1 to vol_0, with its restriction
    swapped to match a measure ordered the other way, reads B on vol_1 and A on
    vol_0."""
    reaction = make_wired_reaction(two_cells, ["B"], ["A"], subdomains=(vol_1, vol_0))
    reaction.restriction = ("-", "+")
    assert np.isclose(rate_on_interface(reaction, two_cells), K_PLUS * 5 - K_MINUS * 2)


@pytest.mark.parametrize("reactants, n", [(["A"], 1), (["A", "A"], 2)])
def test_reactant_residual_is_rate_times_stoichiometry(two_cells, reactants, n):
    """Test that each occurrence of a reactant consumes it at the rate R on its
    side of the interface."""
    reaction = make_wired_reaction(two_cells, reactants, ["B"])
    rate = rate_on_interface(reaction, two_cells)
    F_0, _ = reaction.get_formulation(two_cells.dS, species=[], temperature=None)
    assert np.isclose(residual_on(F_0, vol_0, two_cells), n * rate)


@pytest.mark.parametrize("products, n", [(["B"], 1), (["B", "B"], 2)])
def test_product_residual_is_minus_rate_times_stoichiometry(two_cells, products, n):
    """Test that each occurrence of a product produces it at the rate R on its side
    of the interface."""
    reaction = make_wired_reaction(two_cells, ["A"], products)
    rate = rate_on_interface(reaction, two_cells)
    _, F_1 = reaction.get_formulation(two_cells.dS, species=[], temperature=None)
    assert np.isclose(residual_on(F_1, vol_1, two_cells), -n * rate)


def test_reactant_residual_is_zero_on_the_other_side(two_cells):
    """Test that the reactant term only acts on the cell of the first subdomain."""
    reaction = make_wired_reaction(two_cells, ["A"], ["B"])
    F_0, _ = reaction.get_formulation(two_cells.dS, species=[], temperature=None)
    assert residual_on(F_0, vol_1, two_cells) == 0


def test_irreversible_reaction_adds_nothing_to_second_subdomain(two_cells):
    """Test that a reaction without products returns a zero form for vol_1."""
    reaction = make_wired_reaction(two_cells, ["A"], [])
    _, F_1 = reaction.get_formulation(two_cells.dS, species=[], temperature=None)
    assert F_1 == 0
