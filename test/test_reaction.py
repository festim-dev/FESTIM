from mpi4py import MPI

import numpy as np
import pytest
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import create_unit_cube
from ufl import exp

import festim as F

my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=None)


def convert_rates(reaction, function_space, temperature):
    """Convert a reaction's rate coefficients to fenics objects, as the
    HydrogenTransportProblem does before building the reaction term."""
    for rate in reaction.rate_coefficients:
        if rate.input_value is not None:
            rate.convert_input_value(
                function_space=function_space,
                temperature=temperature,
                up_to_ufl_expr=True,
            )


def test_reaction_is_deprecated_alias():
    """Test that F.Reaction emits a DeprecationWarning pointing at
    ArrheniusReaction."""
    # BUILD / RUN / TEST
    with pytest.warns(DeprecationWarning, match="ArrheniusReaction"):
        reaction = F.Reaction(reactant=F.Species("A"), k_0=1.0, E_k=0.2, volume=my_vol)
    assert isinstance(reaction, F.ArrheniusReaction)


def test_generic_reaction_is_a_reaction_base():
    """Test that GenericReaction is a subclass of ReactionBase."""
    assert issubclass(F.GenericReaction, F.ReactionBase)


def test_reaction_base_str():
    """Test __str__ shows consumed species on the left and produced on the right."""
    # BUILD
    A, B = F.Species("A"), F.Species("B")
    reaction = F.ReactionBase(reaction_rate=1.0, reactant=A, product=B, volume=my_vol)

    # RUN / TEST
    assert str(reaction) == "A --> B"


def test_reaction_base_str_no_produced_species():
    """Test __str__ shows a one-way arrow with no produced species."""
    # BUILD
    reaction = F.ReactionBase(reaction_rate=1.0, reactant=F.Species("A"), volume=my_vol)

    # RUN / TEST
    assert str(reaction) == "A -->"


def test_reaction_base_str_with_repeated_species():
    """Test __str__ lists a repeated species once per appearance."""
    # BUILD
    A, B = F.Species("A"), F.Species("B")
    reaction = F.ReactionBase(
        reaction_rate=1.0, reactant=[A, A], product=[B, B, B], volume=my_vol
    )

    # RUN / TEST
    assert str(reaction) == "A + A --> B + B + B"


def test_reaction_base_repr():
    """Test __repr__ shows the reaction and its reaction rate."""
    # BUILD
    A, B = F.Species("A"), F.Species("B")
    reaction = F.ReactionBase(reaction_rate=2.0, reactant=A, product=B, volume=my_vol)

    # RUN / TEST
    assert repr(reaction) == "ReactionBase(A --> B, 2.0)"


def test_reaction_base_rate_coefficients():
    """Test that a ReactionBase exposes its single reaction_rate as its rate
    coefficient."""
    # BUILD
    reaction = F.ReactionBase(reaction_rate=1.0, reactant=F.Species("A"), volume=my_vol)

    # RUN / TEST
    assert reaction.rate_coefficients == [reaction.reaction_rate]


def test_reaction_base_species_lists_involved_species():
    """Test that the species property lists the stoichiometry species and the rate's
    concentration-dependence species."""
    # BUILD
    A, B, C = F.Species("A"), F.Species("B"), F.Species("C")
    reaction = F.ReactionBase(
        reaction_rate=lambda c_C: c_C,
        reactant=A,
        product=B,
        volume=my_vol,
        arg_to_species={"c_C": C},
    )

    # RUN / TEST
    assert reaction.species == [A, B, C]


def test_reaction_base_reaction_term_is_the_rate():
    """Test that reaction_term returns the reaction rate itself (not multiplied by
    any concentration)."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)
    V = functionspace(mesh, ("Lagrange", 1))
    reaction = F.ReactionBase(reaction_rate=3.0, reactant=F.Species("A"), volume=my_vol)

    # RUN
    convert_rates(reaction, V, temperature=500.0)

    # TEST
    assert reaction.reaction_term() is reaction.reaction_rate.fenics_object


def test_reaction_base_reaction_term_before_conversion_raises_error():
    """Test that reaction_term fails clearly when the rate has not been converted to
    a fenics object yet."""
    # BUILD
    reaction = F.ReactionBase(reaction_rate=3.0, reactant=F.Species("A"), volume=my_vol)

    # RUN / TEST
    with pytest.raises(AssertionError, match="fenics_object has not been defined"):
        reaction.reaction_term()


def test_reaction_base_arbitrary_rate_depends_on_concentrations():
    """Test that a ReactionBase rate can be an arbitrary function of species
    concentrations, e.g. R = k * (c_A - c_B)."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)
    V = functionspace(mesh, ("Lagrange", 1))
    A, B = F.Species("A"), F.Species("B")
    A.solution = Function(V)
    B.solution = Function(V)
    reaction = F.ReactionBase(
        reaction_rate=lambda c_A, c_B: 2.0 * (c_A - c_B),
        reactant=A,
        product=B,
        volume=my_vol,
        arg_to_species={"c_A": A, "c_B": B},
    )

    # RUN
    convert_rates(reaction, V, temperature=500.0)

    # TEST
    expected = 2.0 * (A.concentration - B.concentration)
    assert reaction.reaction_term() == expected


def test_reaction_base_create_sources_count():
    """Test a ReactionBase expands into one source per involved species."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = functionspace(mesh, ("Lagrange", 1))
    A, B = F.Species("A"), F.Species("B")
    reaction = F.ReactionBase(reaction_rate=3.0, reactant=A, product=B, volume=my_vol)
    convert_rates(reaction, V, temperature=500.0)

    # RUN
    sources = reaction.create_sources()

    # TEST
    assert len(sources) == 2


def test_reaction_base_consumed_species_source():
    """Test that a species with a negative coefficient gets a sink (source -R)."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = functionspace(mesh, ("Lagrange", 1))
    A, B = F.Species("A"), F.Species("B")
    reaction = F.ReactionBase(reaction_rate=3.0, reactant=A, product=B, volume=my_vol)
    convert_rates(reaction, V, temperature=500.0)

    # RUN
    sink = next(s for s in reaction.create_sources() if s.species is A)

    # TEST
    assert sink.value.input_value == -reaction.reaction_term()


def test_reaction_base_produced_species_source():
    """Test that a species with a positive coefficient gets a source (+R)."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = functionspace(mesh, ("Lagrange", 1))
    A, B = F.Species("A"), F.Species("B")
    reaction = F.ReactionBase(reaction_rate=3.0, reactant=A, product=B, volume=my_vol)
    convert_rates(reaction, V, temperature=500.0)

    # RUN
    source = next(s for s in reaction.create_sources() if s.species is B)

    # TEST
    assert source.value.input_value == reaction.reaction_term()


@pytest.mark.parametrize("nb_occurences", [2, 3])
def test_reaction_base_repeated_reactant_gets_one_sink_per_appearance(nb_occurences):
    """Test that a species listed several times as a reactant gets as many sinks,
    which add up to a consumption of nb_occurences * R."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = functionspace(mesh, ("Lagrange", 1))
    A = F.Species("A")
    reaction = F.ReactionBase(
        reaction_rate=3.0, reactant=[A] * nb_occurences, volume=my_vol
    )
    convert_rates(reaction, V, temperature=500.0)

    # RUN
    sinks = [s for s in reaction.create_sources() if s.species is A]

    # TEST
    assert [s.value.input_value for s in sinks] == [-reaction.reaction_term()] * (
        nb_occurences
    )


@pytest.mark.parametrize("nb_occurences", [2, 3])
def test_reaction_base_repeated_product_gets_one_source_per_appearance(nb_occurences):
    """Test that a species listed several times as a product gets as many sources,
    which add up to a production of nb_occurences * R."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = functionspace(mesh, ("Lagrange", 1))
    A, B = F.Species("A"), F.Species("B")
    reaction = F.ReactionBase(
        reaction_rate=3.0, reactant=A, product=[B] * nb_occurences, volume=my_vol
    )
    convert_rates(reaction, V, temperature=500.0)

    # RUN
    sources = [s for s in reaction.create_sources() if s.species is B]

    # TEST
    assert [s.value.input_value for s in sources] == [reaction.reaction_term()] * (
        nb_occurences
    )


def test_reaction_base_species_on_both_sides_gets_a_sink_and_a_source():
    """Test that a species that is both reactant and product gets both terms, which
    cancel out."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = functionspace(mesh, ("Lagrange", 1))
    A, B = F.Species("A"), F.Species("B")
    reaction = F.ReactionBase(
        reaction_rate=3.0, reactant=[A, B], product=A, volume=my_vol
    )
    convert_rates(reaction, V, temperature=500.0)

    # RUN
    terms = [s.value.input_value for s in reaction.create_sources() if s.species is A]

    # TEST
    assert terms == [-reaction.reaction_term(), reaction.reaction_term()]


def test_reaction_base_implicit_species_gets_no_source():
    """Test that an implicit reactant, which has no governing equation, is skipped
    by create_sources."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = functionspace(mesh, ("Lagrange", 1))
    A, B = F.Species("A"), F.Species("B")
    empty = F.ImplicitSpecies(n=1.0, others=[A], name="empty")
    reaction = F.ReactionBase(
        reaction_rate=3.0, reactant=[A, empty], product=B, volume=my_vol
    )
    convert_rates(reaction, V, temperature=500.0)

    # RUN
    sources = reaction.create_sources()

    # TEST
    assert [source.species for source in sources] == [A, B]


def test_reaction_init():
    """Test that the Reaction class initialises correctly."""
    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create a product species
    product = F.Species("C")

    # create a reaction between the two species
    reaction = F.ArrheniusReaction(
        reactant=[species1, species2],
        product=product,
        k_0=1.0,
        E_k=0.2,
        p_0=0.1,
        E_p=0.3,
        volume=my_vol,
    )

    # check that the attributes are set correctly
    assert reaction.reactant == [species1, species2]
    assert reaction.product == [product]
    assert reaction.k_0 == 1.0
    assert reaction.E_k == 0.2
    assert reaction.p_0 == 0.1
    assert reaction.E_p == 0.3


def test_reaction_repr():
    """Test that the Reaction __repr__ method returns the expected string."""

    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create a product species
    product = F.Species("C")

    # create a reaction between the two species
    reaction = F.ArrheniusReaction(
        reactant=[species1, species2],
        product=product,
        k_0=1.0,
        E_k=0.2,
        p_0=0.1,
        E_p=0.3,
        volume=my_vol,
    )

    # check that the __repr__ method returns the expected string
    expected_repr = "ArrheniusReaction(A + B <--> C, 1.0, 0.2, 0.1, 0.3)"
    assert repr(reaction) == expected_repr


def test_reaction_repr_2_products():
    """Test that the Reaction __repr__ method returns the expected string."""

    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create two product species
    product1 = F.Species("C")
    product2 = F.Species("D")

    # create a reaction between the two species
    reaction = F.ArrheniusReaction(
        reactant=[species1, species2],
        product=[product1, product2],
        k_0=1.0,
        E_k=0.2,
        p_0=0.1,
        E_p=0.3,
        volume=my_vol,
    )

    # check that the __repr__ method returns the expected string
    expected_repr = "ArrheniusReaction(A + B <--> C + D, 1.0, 0.2, 0.1, 0.3)"
    assert repr(reaction) == expected_repr


def test_reaction_repr_0_products():
    """Test that the Reaction __repr__ method returns the expected string."""

    # create two species
    species1 = F.Species("A")

    # create a reaction between the two species
    reaction = F.ArrheniusReaction(
        reactant=species1,
        k_0=1.0,
        E_k=0.2,
        volume=my_vol,
    )

    # check that the __repr__ method returns the expected string
    expected_repr = "ArrheniusReaction(A -->, 1.0, 0.2, None, None)"
    assert repr(reaction) == expected_repr


def test_reaction_str():
    """Test that the Reaction __str__ method returns the expected string."""

    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create a product species
    product = F.Species("C")

    # create a reaction between the two species
    reaction = F.ArrheniusReaction(
        reactant=[species1, species2],
        product=product,
        k_0=1.0,
        E_k=0.2,
        p_0=0.1,
        E_p=0.3,
        volume=my_vol,
    )

    # check that the __str__ method returns the expected string
    expected_str = "A + B <--> C"
    assert str(reaction) == expected_str


def test_reaction_str_2_products():
    """Test that the Reaction __str__ method returns the expected string when there are
    2 products."""

    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create a product species
    product1 = F.Species("C")
    product2 = F.Species("D")

    # create a reaction between the two species
    reaction = F.ArrheniusReaction(
        reactant=[species1, species2],
        product=[product1, product2],
        k_0=1.0,
        E_k=0.2,
        p_0=0.1,
        E_p=0.3,
        volume=my_vol,
    )

    # check that the __str__ method returns the expected string
    expected_str = "A + B <--> C + D"
    assert str(reaction) == expected_str


def test_reaction_str_no_products():
    """Test that the Reaction __str__ method returns the expected string when there are
    2 products."""

    # create two species
    species1 = F.Species("A")

    # create a reaction between the two species
    reaction = F.ArrheniusReaction(
        reactant=species1,
        k_0=1.0,
        E_k=0.2,
        volume=my_vol,
    )

    # check that the __str__ method returns the expected string
    expected_str = "A -->"
    assert str(reaction) == expected_str


@pytest.mark.parametrize("temperature", [300.0, 350, 370, 500.0])
def test_reaction_reaction_term(temperature):
    """Test that the Reaction.reaction_term method returns the expected reaction
    term."""

    mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = functionspace(mesh, ("Lagrange", 1))

    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")
    species1.solution = Function(V)
    species2.solution = Function(V)

    # create a product species
    product = F.Species("C")
    product.solution = Function(V)

    # create a reaction between the two species
    reaction = F.ArrheniusReaction(
        reactant=[species1, species2],
        product=product,
        k_0=1.0,
        E_k=0.2,
        p_0=0.1,
        E_p=0.3,
        volume=my_vol,
    )

    # test the reaction term at a given temperature
    def arrhenius(pre, act, T):
        return pre * exp(-act / (F.k_B * T))

    k = arrhenius(reaction.k_0, reaction.E_k, temperature)
    p = arrhenius(reaction.p_0, reaction.E_p, temperature)

    expected_reaction_term = (
        k * (species1.solution * species2.solution) - p * product.solution
    )

    convert_rates(reaction, V, temperature)
    assert reaction.reaction_term() == expected_reaction_term


@pytest.mark.parametrize("temperature", [300.0, 350, 370, 500.0])
def test_reaction_reaction_term_no_products(temperature):
    mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = functionspace(mesh, ("Lagrange", 1))

    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")
    species1.solution = Function(V)
    species2.solution = Function(V)

    # create a reaction between the two species
    reaction = F.ArrheniusReaction(
        reactant=[species1, species2],
        k_0=1.0,
        E_k=0.2,
        volume=my_vol,
    )

    # test the reaction term at a given temperature
    def arrhenius(pre, act, T):
        return pre * exp(-act / (F.k_B * T))

    k = arrhenius(reaction.k_0, reaction.E_k, temperature)

    expected_reaction_term = k * (species1.solution * species2.solution)

    convert_rates(reaction, V, temperature)
    assert reaction.reaction_term() == expected_reaction_term


@pytest.mark.parametrize("temperature", [300.0, 350, 370, 500.0])
def test_reaction_reaction_term_2_products(temperature):
    """Test that the Reaction.reaction_term method returns the expected reaction term
    with two products."""

    mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = functionspace(mesh, ("Lagrange", 1))

    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")
    species1.solution = Function(V)
    species2.solution = Function(V)

    # create a product species
    product1 = F.Species("C")
    product2 = F.Species("D")
    product1.solution = Function(V)
    product2.solution = Function(V)

    # create a reaction between the two species
    reaction = F.ArrheniusReaction(
        reactant=[species1, species2],
        product=[product1, product2],
        k_0=1.0,
        E_k=0.2,
        p_0=0.1,
        E_p=0.3,
        volume=my_vol,
    )

    # test the reaction term at a given temperature
    def arrhenius(pre, act, T):
        return pre * exp(-act / (F.k_B * T))

    k = arrhenius(reaction.k_0, reaction.E_k, temperature)
    p = arrhenius(reaction.p_0, reaction.E_p, temperature)

    product_of_products = product1.solution * product2.solution
    expected_reaction_term = (
        k * (species1.solution * species2.solution) - p * product_of_products
    )
    convert_rates(reaction, V, temperature)
    assert reaction.reaction_term() == expected_reaction_term


def test_reactant_setter_raises_error_with_zero_length_list():
    """Test a value error is raised when the first reactant is given a wrong type."""
    with pytest.raises(
        ValueError,
        match=(
            r"reactant must be an entry of one or more species objects, not an empty "
            r"list."
        ),
    ):
        F.ArrheniusReaction(
            reactant=[],
            k_0=1,
            E_k=0.1,
            p_0=2,
            E_p=0.2,
            volume=my_vol,
        )


def test_reactant_setter_raises_error_with_wrong_type():
    """Test a type error is raised when the first reactant is given a wrong type."""
    with pytest.raises(
        TypeError,
        match=r"reactant must be an F.Species or F.ImplicitSpecies, not <class 'str'>",
    ):
        F.ArrheniusReaction(
            reactant=["A", F.Species("B")],
            product=F.Species("C"),
            k_0=1,
            E_k=0.1,
            p_0=2,
            E_p=0.2,
            volume=my_vol,
        )


C, D = F.Species("C"), F.Species("D")


@pytest.mark.parametrize(
    "product, expected",
    [(C, [C]), ([C, D], [C, D]), (None, []), ([], [])],
)
def test_product_setter_normalises_to_list(product, expected):
    """Test that the product setter stores the product(s) as a list, accepting a
    single species, a list, None or an empty list."""
    # BUILD / RUN
    reaction = F.GenericReaction(
        reactant=F.Species("A"),
        product=product,
        forward_rate=1.0,
        volume=my_vol,
    )

    # TEST
    assert reaction.product == expected


@pytest.mark.parametrize("product", ["C", 3, [F.Species("C"), "D"]])
def test_product_setter_raises_error_with_wrong_type(product):
    """Test a type error is raised when a product is given a wrong type."""
    with pytest.raises(
        TypeError,
        match=r"product must be an F.Species, a list of F.Species, or None",
    ):
        F.GenericReaction(
            reactant=F.Species("A"),
            product=product,
            forward_rate=1.0,
            volume=my_vol,
        )


def test_arg_to_species_is_stored():
    """Test that a valid arg_to_species mapping is stored on the reaction."""
    # BUILD
    A, C = F.Species("A"), F.Species("C")

    # RUN
    reaction = F.GenericReaction(
        reactant=A,
        product=C,
        forward_rate=1.0,
        backward_rate=lambda c_C: c_C,
        arg_to_species={"c_C": C},
        volume=my_vol,
    )

    # TEST
    assert reaction.arg_to_species == {"c_C": C}


def test_arg_to_species_is_passed_to_rate_coefficient():
    """Test that arg_to_species is handed to the festim.Value of a rate
    coefficient, so the coefficient can depend on species concentrations."""
    # BUILD
    A, C = F.Species("A"), F.Species("C")

    # RUN
    reaction = F.GenericReaction(
        reactant=A,
        product=C,
        forward_rate=lambda T: T,
        backward_rate=lambda c_C: c_C,
        arg_to_species={"c_C": C},
        volume=my_vol,
    )

    # TEST
    assert reaction.backward_rate.species_dependent_value == {"c_C": C}


def test_arg_to_species_raises_error_with_non_species_value():
    """Test a type error is raised when an arg_to_species value is not a Species."""
    with pytest.raises(
        TypeError,
        match=r"arg_to_species values must be a festim.Species",
    ):
        F.GenericReaction(
            reactant=F.Species("A"),
            product=F.Species("C"),
            forward_rate=lambda c: c,
            arg_to_species={"c": "not_a_species"},
            volume=my_vol,
        )


def test_arg_to_species_raises_error_when_rate_depends_on_unmapped_species():
    """Test a value error is raised when a rate coefficient has a species argument
    that is not given in arg_to_species."""
    with pytest.raises(
        ValueError,
        match=r"depend on \['c_C'\], which must be given in arg_to_species",
    ):
        F.GenericReaction(
            reactant=F.Species("A"),
            product=F.Species("C"),
            forward_rate=lambda c_C: c_C,
            volume=my_vol,
        )


def test_arg_to_species_warns_with_orphan_mapping():
    """Test a warning is raised when an arg_to_species key is not a species
    argument of any rate coefficient (the extra key is harmless and ignored)."""
    with pytest.warns(
        UserWarning,
        match=r"arg_to_species contains \['c_C'\], which is not an argument",
    ):
        F.GenericReaction(
            reactant=F.Species("A"),
            product=F.Species("C"),
            forward_rate=lambda T: T,
            arg_to_species={"c_C": F.Species("C")},
            volume=my_vol,
        )


def test_generic_reaction_term():
    """Test that GenericReaction.reaction_term returns the net mass-action rate
    directly (i.e. without going through the ArrheniusReaction subclass)."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)
    V = functionspace(mesh, ("Lagrange", 1))
    A, B, C = F.Species("A"), F.Species("B"), F.Species("C")
    for spe in (A, B, C):
        spe.solution = Function(V)

    reaction = F.GenericReaction(
        reactant=[A, B],
        product=C,
        forward_rate=2.0,
        backward_rate=3.0,
        volume=my_vol,
    )

    # RUN
    convert_rates(reaction, V, temperature=500.0)

    # TEST
    k1 = reaction.forward_rate.fenics_object
    k2 = reaction.backward_rate.fenics_object
    expected = k1 * (A.solution * B.solution) - k2 * C.solution
    assert reaction.reaction_term() == expected


def test_rate_value_species_mapping_is_honored():
    """Test that a species mapping given directly on a rate coefficient Value is
    honored (exposed through arg_to_species) without also passing arg_to_species."""
    # BUILD
    A, C = F.Species("A"), F.Species("C")

    # RUN
    reaction = F.GenericReaction(
        reactant=A,
        product=C,
        forward_rate=F.Value(lambda c_C: c_C, species_dependent_value={"c_C": C}),
        volume=my_vol,
    )

    # TEST
    assert reaction.arg_to_species == {"c_C": C}


def test_arg_to_species_raises_error_when_mapping_given_twice():
    """Test a value error is raised when a species mapping is given both on a rate
    coefficient Value and via arg_to_species."""
    A, C = F.Species("A"), F.Species("C")
    with pytest.raises(
        ValueError,
        match=r"provide it in only one place",
    ):
        F.GenericReaction(
            reactant=A,
            product=C,
            forward_rate=F.Value(lambda c_C: c_C, species_dependent_value={"c_C": C}),
            arg_to_species={"c_C": C},
            volume=my_vol,
        )


def test_p_0_setter_raises_error_with_no_product():
    """Test p_0 must be None when there is no product."""
    with pytest.raises(
        ValueError,
        match=r"p_0 must be None, not 2 when no products are present.",
    ):
        F.ArrheniusReaction(
            reactant=[F.Species("A")],
            k_0=1,
            E_k=0.1,
            p_0=2,
            volume=my_vol,
        )


def test_E_p_setter_raises_error_when_none_with_product():
    """Test E_p cannot be None when there is a product."""
    with pytest.raises(
        ValueError,
        match=r"E_p cannot be None when reaction products are present.",
    ):
        F.ArrheniusReaction(
            reactant=[F.Species("A")],
            product=[F.Species("C")],
            k_0=1,
            E_k=0.1,
            p_0=0.1,
            volume=my_vol,
        )


def test_p_0_setter_raises_error_when_none_with_product():
    """Test p_0 cannot be None when there is a product."""
    with pytest.raises(
        ValueError,
        match=r"p_0 cannot be None when reaction products are present.",
    ):
        F.ArrheniusReaction(
            reactant=[F.Species("A")],
            product=[F.Species("C")],
            k_0=1,
            E_k=0.1,
            E_p=1,
            volume=my_vol,
        )


def test_E_p_setter_raises_error_with_no_product():
    """Test E_p must be None when there is no product."""
    with pytest.raises(
        ValueError,
        match=r"E_p must be None, not 2 when no products are present.",
    ):
        F.ArrheniusReaction(
            reactant=[F.Species("A")],
            k_0=1,
            E_k=0.1,
            E_p=2,
            volume=my_vol,
        )


# BUILD
mat = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)

vol1 = F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=mat)
vol2 = F.VolumeSubdomain1D(id=2, borders=[0.5, 1], material=mat)
my_model = F.HydrogenTransportProblemDiscontinuous()
my_model.mesh = F.Mesh1D(np.linspace(0, 1, 101))

my_model.subdomains = [vol1, vol2]

spe1 = F.Species("spe1", subdomains=my_model.volume_subdomains)
spe2 = F.Species("spe2", subdomains=my_model.volume_subdomains)
spe3 = F.Species("spe2", subdomains=my_model.volume_subdomains)
spe4 = F.Species("spe4", mobile=False, subdomains=my_model.volume_subdomains)
spe5 = F.Species("spe5", subdomains=my_model.volume_subdomains)
empty_traps = F.ImplicitSpecies(n=1, others=[spe4], name="implicit_species")

my_model.species = [spe1, spe2, spe3, spe4, spe5]

reac1 = F.ArrheniusReaction(
    reactant=[empty_traps, spe1], product=[], k_0=1, E_k=0, volume=vol1
)
reac2 = F.ArrheniusReaction(
    reactant=[empty_traps, spe2],
    product=[spe5],
    k_0=1,
    E_k=0,
    p_0=1,
    E_p=0,
    volume=vol2,
)

my_model.define_meshtags_and_measures()
for subdomain in my_model.volume_subdomains:
    subdomain.create_subdomain(my_model.mesh.mesh, my_model.volume_meshtags)
    subdomain.transfer_meshtag(my_model.mesh.mesh, my_model.facet_meshtags)

for subdomain in my_model.volume_subdomains:
    my_model.define_function_spaces(subdomain)


def test_decay_reaction_str():
    """Test __str__ shows a one-way decay arrow to the product."""
    # BUILD
    T, He = F.Species("T"), F.Species("He")
    reaction = F.DecayReaction(reactant=T, half_life=10.0, volume=my_vol, product=He)

    # RUN / TEST
    assert str(reaction) == "T --> He"


def test_decay_reaction_str_no_product():
    """Test __str__ shows a one-way decay arrow when no product is tracked."""
    # BUILD
    reaction = F.DecayReaction(reactant=F.Species("T"), half_life=10.0, volume=my_vol)

    # RUN / TEST
    assert str(reaction) == "T -->"


def test_decay_reaction_repr():
    """Test __repr__ includes the half_life."""
    # BUILD
    reaction = F.DecayReaction(reactant=F.Species("T"), half_life=10.0, volume=my_vol)

    # RUN / TEST
    assert repr(reaction) == "DecayReaction(T, half_life=10.0)"


@pytest.mark.parametrize("half_life", [1.0, 10.0, 3.888e8, 12])
def test_decay_reaction_forward_rate_is_decay_constant(half_life):
    """Test the forward rate is the decay constant lambda = ln(2) / half_life."""
    # BUILD
    reaction = F.DecayReaction(
        reactant=F.Species("T"), half_life=half_life, volume=my_vol
    )

    # RUN / TEST
    expected = np.log(2) / half_life
    assert np.isclose(float(reaction.forward_rate.input_value), expected)


def test_decay_reaction_forward_rate_updates_with_half_life():
    """Test that reassigning half_life keeps the forward rate (decay constant) in
    sync."""
    # BUILD
    reaction = F.DecayReaction(reactant=F.Species("T"), half_life=10.0, volume=my_vol)

    # RUN
    reaction.half_life = 20.0

    # TEST
    assert np.isclose(float(reaction.forward_rate.input_value), np.log(2) / 20.0)


def test_decay_reaction_has_no_backward_rate():
    """Test that a decay reaction is irreversible (no backward rate)."""
    # BUILD
    reaction = F.DecayReaction(reactant=F.Species("T"), half_life=10.0, volume=my_vol)

    # RUN / TEST
    assert reaction.backward_rate.input_value is None


def test_decay_reaction_product_is_stored():
    """Test that a given product is stored as a list."""
    # BUILD
    He = F.Species("He")
    reaction = F.DecayReaction(
        reactant=F.Species("T"), half_life=10.0, volume=my_vol, product=He
    )

    # RUN / TEST
    assert reaction.product == [He]


def test_decay_reaction_product_empty_when_none():
    """Test that the product list is empty when no product is given."""
    # BUILD
    reaction = F.DecayReaction(reactant=F.Species("T"), half_life=10.0, volume=my_vol)

    # RUN / TEST
    assert reaction.product == []


@pytest.mark.parametrize("nb_reactants", [2, 3])
def test_decay_reaction_raises_error_with_several_reactants(nb_reactants):
    """Test that a decay reaction accepts a single reactant only."""
    # BUILD
    reactants = [F.Species(f"A{i}") for i in range(nb_reactants)]

    # RUN / TEST
    with pytest.raises(
        ValueError, match="reactant must be a single species for a DecayReaction"
    ):
        F.DecayReaction(reactant=reactants, half_life=10.0, volume=my_vol)


def test_decay_reaction_accepts_a_list_of_one_reactant():
    """Test that a single reactant given as a list is accepted."""
    # BUILD
    T = F.Species("T")

    # RUN
    reaction = F.DecayReaction(reactant=[T], half_life=10.0, volume=my_vol)

    # TEST
    assert reaction.reactant == [T]


@pytest.mark.parametrize("value", ["abc", lambda t: t, None, True, [1.0]])
def test_half_life_setter_raises_type_error(value):
    """Test that half_life must be a float."""
    # BUILD / RUN / TEST
    with pytest.raises(TypeError, match="half_life must be a float"):
        F.DecayReaction(reactant=F.Species("T"), half_life=value, volume=my_vol)


@pytest.mark.parametrize("value", [0, -1.0, -5])
def test_half_life_setter_raises_value_error(value):
    """Test that half_life must be positive."""
    # BUILD / RUN / TEST
    with pytest.raises(ValueError, match="half_life must be positive"):
        F.DecayReaction(reactant=F.Species("T"), half_life=value, volume=my_vol)


def test_decay_reaction_term():
    """Test the net rate of a decay reaction is lambda * c (no backward term)."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)
    V = functionspace(mesh, ("Lagrange", 1))
    T = F.Species("T")
    T.solution = Function(V)
    reaction = F.DecayReaction(reactant=T, half_life=10.0, volume=my_vol)

    # RUN
    convert_rates(reaction, V, temperature=500.0)

    # TEST
    expected = reaction.forward_rate.fenics_object * T.solution
    assert reaction.reaction_term() == expected


def test_decay_reaction_create_sources_count_with_product():
    """Test that a decay with a product expands into two sources (sink + source)."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = functionspace(mesh, ("Lagrange", 1))
    T, He = F.Species("T"), F.Species("He")
    T.solution = Function(V)
    He.solution = Function(V)
    reaction = F.DecayReaction(reactant=T, half_life=10.0, volume=my_vol, product=He)
    convert_rates(reaction, V, temperature=500.0)

    # RUN
    sources = reaction.create_sources()

    # TEST
    assert len(sources) == 2


def test_decay_reaction_reactant_is_consumed():
    """Test that the decaying reactant is a sink (source value -R)."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = functionspace(mesh, ("Lagrange", 1))
    T, He = F.Species("T"), F.Species("He")
    T.solution = Function(V)
    He.solution = Function(V)
    reaction = F.DecayReaction(reactant=T, half_life=10.0, volume=my_vol, product=He)
    convert_rates(reaction, V, temperature=500.0)

    # RUN
    sink = next(s for s in reaction.create_sources() if s.species is T)

    # TEST
    assert sink.value.input_value == -reaction.reaction_term()


def test_decay_reaction_product_is_produced():
    """Test that the decay product is a source (source value +R)."""
    # BUILD
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = functionspace(mesh, ("Lagrange", 1))
    T, He = F.Species("T"), F.Species("He")
    T.solution = Function(V)
    He.solution = Function(V)
    reaction = F.DecayReaction(reactant=T, half_life=10.0, volume=my_vol, product=He)
    convert_rates(reaction, V, temperature=500.0)

    # RUN
    source = next(s for s in reaction.create_sources() if s.species is He)

    # TEST
    assert source.value.input_value == reaction.reaction_term()
