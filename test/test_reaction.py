from mpi4py import MPI

import pytest
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import create_unit_cube
from ufl import exp

import festim as F

my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=None)


def test_reaction_init():
    """Test that the Reaction class initialises correctly"""
    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create a product species
    product = F.Species("C")

    # create a reaction between the two species
    reaction = F.Reaction(
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
    assert reaction.product == product
    assert reaction.k_0 == 1.0
    assert reaction.E_k == 0.2
    assert reaction.p_0 == 0.1
    assert reaction.E_p == 0.3


def test_reaction_repr():
    """Test that the Reaction __repr__ method returns the expected string"""

    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create a product species
    product = F.Species("C")

    # create a reaction between the two species
    reaction = F.Reaction(
        reactant=[species1, species2],
        product=product,
        k_0=1.0,
        E_k=0.2,
        p_0=0.1,
        E_p=0.3,
        volume=my_vol,
    )

    # check that the __repr__ method returns the expected string
    expected_repr = "Reaction(A + B <--> C, 1.0, 0.2, 0.1, 0.3)"
    assert repr(reaction) == expected_repr


def test_reaction_repr_2_products():
    """Test that the Reaction __repr__ method returns the expected string"""

    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create two product species
    product1 = F.Species("C")
    product2 = F.Species("D")

    # create a reaction between the two species
    reaction = F.Reaction(
        reactant=[species1, species2],
        product=[product1, product2],
        k_0=1.0,
        E_k=0.2,
        p_0=0.1,
        E_p=0.3,
        volume=my_vol,
    )

    # check that the __repr__ method returns the expected string
    expected_repr = "Reaction(A + B <--> C + D, 1.0, 0.2, 0.1, 0.3)"
    assert repr(reaction) == expected_repr


def test_reaction_repr_0_products():
    """Test that the Reaction __repr__ method returns the expected string"""

    # create two species
    species1 = F.Species("A")

    # create a reaction between the two species
    reaction = F.Reaction(
        reactant=species1,
        k_0=1.0,
        E_k=0.2,
        volume=my_vol,
    )

    # check that the __repr__ method returns the expected string
    expected_repr = "Reaction(A <--> , 1.0, 0.2, None, None)"
    assert repr(reaction) == expected_repr


def test_reaction_str():
    """Test that the Reaction __str__ method returns the expected string"""

    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create a product species
    product = F.Species("C")

    # create a reaction between the two species
    reaction = F.Reaction(
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
    """Test that the Reaction __str__ method returns the expected string when there are 2 products"""

    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create a product species
    product1 = F.Species("C")
    product2 = F.Species("D")

    # create a reaction between the two species
    reaction = F.Reaction(
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
    """Test that the Reaction __str__ method returns the expected string when there are 2 products"""

    # create two species
    species1 = F.Species("A")

    # create a reaction between the two species
    reaction = F.Reaction(
        reactant=species1,
        k_0=1.0,
        E_k=0.2,
        volume=my_vol,
    )

    # check that the __str__ method returns the expected string
    expected_str = "A <--> "
    assert str(reaction) == expected_str


@pytest.mark.parametrize("temperature", [300.0, 350, 370, 500.0])
def test_reaction_reaction_term(temperature):
    """Test that the Reaction.reaction_term method returns the expected reaction term"""

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
    reaction = F.Reaction(
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

    assert reaction.reaction_term(temperature) == expected_reaction_term


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
    reaction = F.Reaction(
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

    assert reaction.reaction_term(temperature) == expected_reaction_term


@pytest.mark.parametrize("temperature", [300.0, 350, 370, 500.0])
def test_reaction_reaction_term_2_products(temperature):
    """Test that the Reaction.reaction_term method returns the expected reaction term with two products"""

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
    reaction = F.Reaction(
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
    assert reaction.reaction_term(temperature) == expected_reaction_term


def test_reactant_setter_raises_error_with_zero_length_list():
    """Test a value error is raised when the first reactant is given a wrong type."""
    with pytest.raises(
        ValueError,
        match="reactant must be an entry of one or more species objects, not an empty list.",
    ):
        F.Reaction(
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
        match="reactant must be an F.Species or F.ImplicitSpecies, not <class 'str'>",
    ):
        F.Reaction(
            reactant=["A", F.Species("B")],
            product=F.Species("C"),
            k_0=1,
            E_k=0.1,
            p_0=2,
            E_p=0.2,
            volume=my_vol,
        )


def test_product_setter_raise_error_p_0_no_product():
    with pytest.raises(
        ValueError,
        match="p_0 must be None, not 2 when no products are present.",
    ):
        reaction = F.Reaction(
            reactant=[F.Species("A")],
            k_0=1,
            E_k=0.1,
            p_0=2,
            volume=my_vol,
        )
        reaction.reaction_term(temperature=500)


def test_no_E_p_with_product():
    with pytest.raises(
        ValueError,
        match="E_p cannot be None when reaction products are present.",
    ):
        reaction = F.Reaction(
            reactant=[F.Species("A")],
            product=[F.Species("C")],
            k_0=1,
            E_k=0.1,
            p_0=0.1,
            volume=my_vol,
        )
        reaction.reaction_term(temperature=500)


def test_no_p_0_with_product():
    with pytest.raises(
        ValueError,
        match="p_0 cannot be None when reaction products are present.",
    ):
        reaction = F.Reaction(
            reactant=[F.Species("A")],
            product=[F.Species("C")],
            k_0=1,
            E_k=0.1,
            E_p=1,
            volume=my_vol,
        )
        reaction.reaction_term(temperature=500)


def test_product_setter_raise_error_E_p_no_product():
    with pytest.raises(
        ValueError,
        match="E_p must be None, not 2 when no products are present.",
    ):
        reaction = F.Reaction(
            reactant=[F.Species("A")],
            k_0=1,
            E_k=0.1,
            E_p=2,
            volume=my_vol,
        )
        reaction.reaction_term(temperature=500)
