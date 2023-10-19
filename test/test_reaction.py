import pytest
import festim as F
from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import create_unit_cube
from mpi4py import MPI
from ufl import exp


def test_reaction_init():
    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create a product species
    product = F.Species("C")

    # create a reaction between the two species
    reaction = F.Reaction(
        species1, species2, product, k_0=1.0, E_k=0.2, p_0=0.1, E_p=0.3
    )

    # check that the attributes are set correctly
    assert reaction.reactant1 == species1
    assert reaction.reactant2 == species2
    assert reaction.product == product
    assert reaction.k_0 == 1.0
    assert reaction.E_k == 0.2
    assert reaction.p_0 == 0.1
    assert reaction.E_p == 0.3


def test_reaction_repr():
    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create a product species
    product = F.Species("C")

    # create a reaction between the two species
    reaction = F.Reaction(
        species1, species2, product, k_0=1.0, E_k=0.2, p_0=0.1, E_p=0.3
    )

    # check that the __repr__ method returns the expected string
    expected_repr = "Reaction(A + B <--> C, 1.0, 0.2, 0.1, 0.3)"
    assert repr(reaction) == expected_repr


def test_reaction_str():
    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create a product species
    product = F.Species("C")

    # create a reaction between the two species
    reaction = F.Reaction(
        species1, species2, product, k_0=1.0, E_k=0.2, p_0=0.1, E_p=0.3
    )

    # check that the __str__ method returns the expected string
    expected_str = "A + B <--> C"
    assert str(reaction) == expected_str


@pytest.mark.parametrize("temperature", [300.0, 350, 370, 500.0])
def test_reaction_reaction_term(temperature):
    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create a product species
    product = F.Species("C")

    # create a reaction between the two species
    reaction = F.Reaction(
        species1, species2, product, k_0=1.0, E_k=0.2, p_0=0.1, E_p=0.3
    )

    # set the concentrations of the species
    mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    species1.solution = Function(V)
    species2.solution = Function(V)
    product.solution = Function(V)

    # test the reaction term at a given temperature
    def arrhenius(pre, act, T):
        return pre * exp(-act / (F.k_B * T))

    k = arrhenius(reaction.k_0, reaction.E_k, temperature)
    p = arrhenius(reaction.p_0, reaction.E_p, temperature)
    expected_reaction_term = (
        k * species1.solution * species2.solution - p * product.solution
    )

    assert reaction.reaction_term(temperature) == expected_reaction_term
