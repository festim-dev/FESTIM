import numpy as np
import pytest

import festim as F


def test_several_reactions_on_same_interface():
    my_model = F.HydrogenTransportProblemDiscontinuous()

    mat = F.Material(D_0=1, E_D=0)
    vol1 = F.VolumeSubdomain1D(id=1, material=mat, borders=[0, 0.5])
    vol2 = F.VolumeSubdomain1D(id=2, material=mat, borders=[0.5, 1])
    left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0))

    my_model.subdomains = [vol1, vol2, left]
    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 21))

    A = F.Species("A", subdomains=[vol1])
    B = F.Species("B", subdomains=[vol2])
    C = F.Species("C", subdomains=[vol2])
    my_model.species = [A, B, C]

    my_model.interfaces = [
        F.InterfaceReaction(
            id=1,
            subdomains=[vol1, vol2],
            k_plus=1,
            k_minus=1,
            reactants=[A, A],
            products=[B],
        ),
        F.InterfaceReaction(
            id=1,
            subdomains=[vol1, vol2],
            k_plus=1,
            k_minus=1,
            reactants=[A],
            products=[C],
        ),
    ]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(species=A, subdomain=left, value=1),
    ]
    my_model.temperature = 300
    my_model.settings = F.Settings(final_time=1, atol=1e-10, rtol=1e-10, stepsize=0.1)

    my_model.initialise()
    my_model.run()


def build_two_slabs(declarations):
    """Steady 1D problem: A lives on [0, 0.5] with A = 1 at x = 0, B lives on
    [0.5, 1] with B = 0 at x = 1, and D = 1 on both sides.

    Each declaration is ``(forward, k_plus, k_minus)``: a forward reaction is
    declared A -> B with subdomains [vol1, vol2], a backward one B -> A with
    subdomains [vol2, vol1]. All of them share interface id 3.
    """
    mat = F.Material(D_0=1, E_D=0)
    vol1 = F.VolumeSubdomain1D(id=1, material=mat, borders=[0, 0.5])
    vol2 = F.VolumeSubdomain1D(id=2, material=mat, borders=[0.5, 1])
    left = F.SurfaceSubdomain1D(id=4, x=0)
    right = F.SurfaceSubdomain1D(id=5, x=1)
    A = F.Species("A", subdomains=[vol1])
    B = F.Species("B", subdomains=[vol2])

    interfaces = []
    for forward, k_plus, k_minus in declarations:
        sides = ([vol1, vol2], [A], [B]) if forward else ([vol2, vol1], [B], [A])
        interfaces.append(
            F.InterfaceReaction(
                id=3,
                subdomains=sides[0],
                reactants=sides[1],
                products=sides[2],
                k_plus=k_plus,
                k_minus=k_minus,
            )
        )

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh1D(np.linspace(0, 1, 21)),
        subdomains=[vol1, vol2, left, right],
        species=[A, B],
        interfaces=interfaces,
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=left, value=1, species=A),
            F.FixedConcentrationBC(subdomain=right, value=0, species=B),
        ],
        temperature=300,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    return model, (vol1, vol2), (A, B)


def value_at(species, volume, x):
    """Value of the species' solution on ``volume`` at the dof closest to ``x``."""
    u = species.subdomain_to_post_processing_solution[volume]
    dof_x = u.function_space.tabulate_dof_coordinates()[:, 0]
    return u.x.array[np.argmin(np.abs(dof_x - x))]


@pytest.mark.parametrize(
    "declarations",
    [
        [(True, 1, 0.5)],
        [(False, 0.5, 1)],
        [(True, 0, 0), (False, 0.5, 1)],
        [(False, 0, 0), (True, 1, 0.5)],
    ],
    ids=["forward", "backward", "backward-after-forward", "forward-after-backward"],
)
def test_reaction_independent_of_declared_subdomain_order(declarations):
    """Test that R = c_A - 0.5 c_B gives the analytical interface values
    A = 5/7, B = 2/7 whichever way round the reaction, and any other reaction on
    the same id, is declared. A reaction declared the other way round from the
    first one on its id used to contribute nothing."""
    # BUILD
    model, (vol1, vol2), (A, B) = build_two_slabs(declarations)

    # RUN
    model.initialise()
    model.run()

    # TEST
    computed = [value_at(A, vol1, 0.5), value_at(B, vol2, 0.5)]
    assert np.allclose(computed, [5 / 7, 2 / 7])


def test_backward_declared_interface_gets_swapped_restriction():
    """Test that an interface declared with its subdomains the other way round
    from the first interface on its id has its restrictions swapped."""
    # BUILD
    model, _, _ = build_two_slabs([(True, 1, 1), (False, 1, 1)])

    # RUN
    model.initialise()

    # TEST
    assert model.interfaces[1].restriction == ("-", "+")


def test_restriction_is_reset_when_interface_is_reused():
    """Test that an interface swapped in one model is restored to ("+", "-") when
    it is the first interface on its id in another model."""
    # BUILD
    first_model, _, _ = build_two_slabs([(True, 1, 1), (False, 1, 1)])
    first_model.initialise()
    reused = first_model.interfaces[1]
    second_model, _, _ = build_two_slabs([])
    second_model.interfaces = [reused]

    # RUN
    second_model.initialise()

    # TEST
    assert reused.restriction == ("+", "-")


def test_interfaces_sharing_id_on_different_subdomains_raise():
    """Test that two interfaces sharing an id but separating different pairs of
    subdomains raise a ValueError."""
    # BUILD
    model, (vol1, vol2), (A, B) = build_two_slabs([(True, 1, 1)])
    vol3 = F.VolumeSubdomain1D(id=6, material=vol1.material, borders=[1, 1.5])
    model.mesh = F.Mesh1D(np.linspace(0, 1.5, 31))
    model.subdomains = [*model.subdomains, vol3]
    C = F.Species("C", subdomains=[vol3])
    B.subdomains = [vol2, vol3]
    model.species = [A, B, C]
    model.interfaces.append(
        F.InterfaceReaction(
            id=3,
            subdomains=[vol2, vol3],
            k_plus=1,
            k_minus=1,
            reactants=[B],
            products=[C],
        )
    )

    # RUN & TEST
    with pytest.raises(ValueError, match="do not all separate"):
        model.initialise()


@pytest.mark.filterwarnings("error:.*lives on volume subdomain")
def test_reaction_with_one_sided_species_runs_without_warning():
    """Test that a model whose species are one-sided by design, coupled only by an
    InterfaceReaction, does not warn that they are missing a subdomain."""
    model, _, _ = build_two_slabs([(True, 1, 0.5)])
    model.initialise()
    model.run()
