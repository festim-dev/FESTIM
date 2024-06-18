import festim as F
import dolfinx
import ufl
import numpy as np
import pytest
from dolfinx.fem import functionspace, Function, Constant
from dolfinx.mesh import create_unit_cube
from mpi4py import MPI


def test_assign_functions_to_species():
    """Test that checks if the function assign_functions_to_species
    creates the correct attributes for each species
    """

    mesh = F.Mesh1D(
        vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    )
    model = F.HydrogenTransportProblem(
        mesh=mesh,
        species=[
            F.Species(name="H"),
            # F.Species(name="Trap"),
        ],
    )
    model.define_function_spaces()
    model.assign_functions_to_species()

    for spe in model.species:
        assert spe.solution is not None
        assert spe.prev_solution is not None
        assert spe.test_function is not None
        assert isinstance(spe.solution, dolfinx.fem.Function)
        assert isinstance(spe.prev_solution, dolfinx.fem.Function)
        assert isinstance(spe.test_function, ufl.Argument)


def test_species_repr_and_str():
    """Test that the __repr__ and __str__ methods of the Species class returns the
    expected string.
    """
    # create a species
    species = F.Species("A")

    # check that the __repr__ method returns the expected string
    expected_repr = "Species(A)"
    assert repr(species) == expected_repr

    # check that the __str__ method returns the expected string
    expected_str = "A"
    assert str(species) == expected_str


def test_implicit_species_repr_and_str():
    """Test that the __repr__ and __str__ methods of the ImplicitSpecies class
    returns the expected string.
    """
    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create an implicit species that depends on the two species
    implicit_species = F.ImplicitSpecies(3.0, [species1, species2], name="C")

    # check that the __repr__ method returns the expected string
    expected_repr = f"ImplicitSpecies(C, 3.0, {[species1, species2]})"
    assert repr(implicit_species) == expected_repr

    # check that the __str__ method returns the expected string
    expected_str = "C"
    assert str(implicit_species) == expected_str


def test_implicit_species_concentration():
    """Test that the concentration of an implicit species is computed
    correctly.
    """
    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create an implicit species that depends on the two species
    implicit_species = F.ImplicitSpecies(3.0, [species1, species2], name="C")

    # set the solutions of the two species
    mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = functionspace(mesh, ("Lagrange", 1))
    species1.solution = Function(V)
    species2.solution = Function(V)

    implicit_species.convert_n_to_dolfinx(function_space=V, t=Constant(mesh, 0.0))

    # test the concentration of the implicit species
    expected_concentration = implicit_species.n_as_dolfinx - (
        species1.solution + species2.solution
    )
    assert implicit_species.concentration == expected_concentration


def test_implicit_species_concentration_with_no_solution():
    """Test that a ValueError is raised when on of the 'others' species
    has no solution and the concentration of the implicit species is
    requested.
    """
    # create two species
    species1 = F.Species("A")
    species2 = F.Species("B")

    # create an implicit species that depends on the two species
    implicit_species = F.ImplicitSpecies(3.0, [species1, species2], name="C")

    # set the solution of the first species
    mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = functionspace(mesh, ("Lagrange", 1))
    species1.solution = Function(V)

    # test that a ValueError is raised when the second species has no solution
    with pytest.raises(ValueError):
        implicit_species.concentration


def test_create_species_and_reaction():
    """test that the trapped_concentration and trap_reaction attributes
    are correctly set"""

    # BUILD
    my_mobile_species = F.Species("test_mobile")
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=None)
    my_trap = F.Trap(
        name="test",
        mobile_species=my_mobile_species,
        k_0=1,
        E_k=1,
        p_0=1,
        E_p=1,
        n=1,
        volume=my_vol,
    )

    # RUN
    my_trap.create_species_and_reaction()

    # TEST
    assert isinstance(my_trap.trapped_concentration, F.Species)
    assert isinstance(my_trap.reaction, F.Reaction)
