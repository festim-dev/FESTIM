import warnings

import numpy as np
import pytest

import festim as F
from festim.material import SolubilityLaw
from festim.subdomain.interface import interface_condition_term


def build_discontinuous_model(
    interface_method=None, problem_method=None, laws=(None, None)
):
    """A minimal 1D two-subdomain discontinuous model, ready to be initialised.

    Args:
        interface_method: the method set on the interface itself
        problem_method: the deprecated problem-level method_interface, if any
        laws: the solubility law of each subdomain, defaulting to the
            Material default on both sides

    Returns:
        the model and its interface
    """
    my_model = F.HydrogenTransportProblemDiscontinuous()
    # the mesh must have a vertex exactly at the interface (x=1)
    vertices = np.concatenate(
        [np.linspace(0, 1, num=10, endpoint=False), np.linspace(1, 2, num=10)]
    )
    my_model.mesh = F.Mesh1D(vertices=vertices)

    materials = [
        F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
        if law is None
        else F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0, solubility_law=law)
        for law in laws
    ]
    vol_0 = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=materials[0])
    vol_1 = F.VolumeSubdomain1D(id=2, borders=[1, 2], material=materials[1])
    left = F.SurfaceSubdomain1D(id=3, x=0)
    right = F.SurfaceSubdomain1D(id=4, x=2)
    my_model.subdomains = [vol_0, vol_1, left, right]

    kwargs = {} if interface_method is None else {"method": interface_method}
    interface = F.Interface(id=5, subdomains=[vol_0, vol_1], **kwargs)
    my_model.interfaces = [interface]

    if problem_method is not None:
        my_model.method_interface = problem_method

    my_model.species = [F.Species("H", subdomains=[vol_0, vol_1])]
    my_model.temperature = 500
    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    return my_model, interface


@pytest.mark.parametrize(
    "method", [F.InterfaceMethod.nitsche, F.InterfaceMethod.penalty]
)
def test_interface_method_survives_initialise(method):
    """The method set on an interface is not overwritten by initialise()."""
    # BUILD
    my_model, interface = build_discontinuous_model(interface_method=method)

    # RUN
    my_model.initialise()

    # TEST
    assert interface.method == method


def test_no_deprecation_warning_when_method_interface_is_not_set():
    """initialise() does not warn about method_interface unless the user set it."""
    # BUILD
    my_model, _ = build_discontinuous_model()

    # RUN
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        my_model.initialise()

    # TEST
    assert not [w for w in record if "method_interface" in str(w.message)]


@pytest.mark.parametrize(
    "method", [F.InterfaceMethod.nitsche, F.InterfaceMethod.penalty]
)
def test_deprecated_problem_method_interface_still_overrides(method):
    """The deprecated problem-level attribute still sets the method of each
    interface."""
    # BUILD
    my_model, interface = build_discontinuous_model(
        interface_method=F.InterfaceMethod.penalty, problem_method=method
    )

    # RUN
    with pytest.warns(DeprecationWarning, match="method_interface"):
        my_model.initialise()

    # TEST
    assert interface.method == method


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


@pytest.mark.parametrize("laws", [("henry", "sievert"), ("sievert", "henry")])
def test_nitsche_rejects_a_mixed_solubility_law_interface(laws):
    """Nitsche raises rather than silently solving the wrong interface condition.

    Its jump is u_0/K_0 - u_1/K_1, which is the interface condition only when both
    sides share a solubility law. On a mixed interface that is the wrong condition
    and the solve would converge cleanly to a wrong solution, so this has to fail
    loudly instead.
    """
    # BUILD
    my_model, _ = build_discontinuous_model(
        interface_method=F.InterfaceMethod.nitsche, laws=laws
    )

    # RUN + TEST
    with pytest.raises(NotImplementedError, match="different solubility laws"):
        my_model.initialise()


@pytest.mark.parametrize("laws", [("henry", "henry"), ("sievert", "sievert")])
def test_nitsche_accepts_a_single_solubility_law_interface(laws):
    """Nitsche is unaffected when both sides share a solubility law."""
    # BUILD
    my_model, interface = build_discontinuous_model(
        interface_method=F.InterfaceMethod.nitsche, laws=laws
    )

    # RUN
    my_model.initialise()

    # TEST
    assert interface.method == F.InterfaceMethod.nitsche


@pytest.mark.parametrize("laws", [("henry", "sievert"), ("sievert", "henry")])
def test_penalty_accepts_a_mixed_solubility_law_interface(laws):
    """The penalty method does support a mixed interface, and is the way to get one."""
    # BUILD
    my_model, interface = build_discontinuous_model(
        interface_method=F.InterfaceMethod.penalty, laws=laws
    )

    # RUN
    my_model.initialise()

    # TEST
    assert interface.method == F.InterfaceMethod.penalty
