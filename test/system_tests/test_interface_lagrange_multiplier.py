"""The Lagrange multiplier interface method in 2D.

A unit square split at y = 1/2, c = 1 on top and c = 0 at the bottom, insulated
sides. Both materials follow Sieverts' law, K = 3 (bottom) and 2 (top), D = 2 and
1, so equilibrium makes the concentration jump from 3/8 to 1/4 across the
interface, and the flux is 3/2 throughout. The profiles are linear in each half, so
P1 reproduces them exactly: whatever is left is the interface method's.

A convergence (MMS) study of the method lives in the documentation / verification
material; here the method is pinned on cases with known, exact answers.
"""

import warnings

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import festim as F
from festim.subdomain.interface import InterfaceMethod

J_EXACT = 1.5  # flux from the top to the bottom


def build(method="lagrange_multiplier", sides_fixed=False, n=8):
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    model = F.HydrogenTransportProblemDiscontinuous()
    model.mesh = F.Mesh(mesh)

    top = F.VolumeSubdomain(
        id=3,
        material=F.Material(D_0=1, E_D=0, K_S_0=2, E_K_S=0),
        locator=lambda x: x[1] >= 0.5,
    )
    bottom = F.VolumeSubdomain(
        id=4,
        material=F.Material(D_0=2, E_D=0, K_S_0=3, E_K_S=0),
        locator=lambda x: x[1] <= 0.5,
    )
    top_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1.0))
    bottom_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0.0))
    subdomains = [top_surface, bottom_surface, top, bottom]

    H = F.Species("H", subdomains=[bottom, top])
    bcs = [
        F.FixedConcentrationBC(top_surface, value=1.0, species=H),
        F.FixedConcentrationBC(bottom_surface, value=0.0, species=H),
    ]
    if sides_fixed:
        # the exact profile imposed on the left side too, on *both* materials, so
        # that the interface ends on a boundary where both sides are prescribed
        left_bottom = F.SurfaceSubdomain(
            id=6, locator=lambda x: np.isclose(x[0], 0.0) & (x[1] <= 0.5)
        )
        left_top = F.SurfaceSubdomain(
            id=7, locator=lambda x: np.isclose(x[0], 0.0) & (x[1] >= 0.5)
        )
        subdomains += [left_bottom, left_top]
        bcs += [
            F.FixedConcentrationBC(left_bottom, value=lambda x: 0.75 * x[1], species=H),
            F.FixedConcentrationBC(
                left_top, value=lambda x: 0.25 + 1.5 * (x[1] - 0.5), species=H
            ),
        ]

    model.subdomains = subdomains
    model.species = [H]
    model.temperature = 400
    model.interfaces = [F.Interface(5, (bottom, top), method=method)]
    model.boundary_conditions = bcs
    model.settings = F.Settings(atol=1e-12, rtol=1e-12, transient=False)
    model.exports = [
        F.SurfaceFlux(field=H, surface=top_surface),
        F.SurfaceFlux(field=H, surface=bottom_surface),
    ]
    return model


def owned_values(f):
    """The values of ``f`` at the dofs this rank owns, gathered on every rank."""
    n = f.function_space.dofmap.index_map.size_local
    return np.concatenate(MPI.COMM_WORLD.allgather(f.x.array[:n]))


@pytest.mark.parametrize(
    "name",
    ["lagrange_multiplier", "Lagrange", "lm", InterfaceMethod.lagrange_multiplier],
)
def test_method_can_be_given_by_name(name):
    material = F.Material(D_0=1, E_D=0)
    volumes = (F.VolumeSubdomain(1, material), F.VolumeSubdomain(2, material))
    interface = F.Interface(5, volumes, method=name)
    assert interface.method == InterfaceMethod.lagrange_multiplier


def test_flux_is_exact_and_is_the_multiplier():
    model = build()
    model.initialise()
    model.run()

    flux_top, flux_bottom = (export.data[-1] for export in model.exports)
    assert np.isclose(flux_top, -J_EXACT, rtol=1e-10)
    assert np.isclose(flux_bottom, J_EXACT, rtol=1e-10)

    # the multiplier is the flux leaving the first subdomain (bottom) into the second
    # one (top): -3/2 at every point of the interface
    interface = model.interfaces[0]
    lam = interface.multipliers[model.species[0]]
    assert np.allclose(owned_values(lam), -J_EXACT, rtol=1e-10)


def test_multiplier_is_pinned_where_both_sides_are_prescribed():
    """At the two ends of the interface on the left side, both concentrations are
    prescribed and the multiplier is set to zero there, with a warning. The
    concentrations, and hence the fluxes through the walls, are still exact."""
    model = build(sides_fixed=True)
    with pytest.warns(UserWarning, match="fixed on both sides"):
        model.initialise()
    model.run()

    flux_top, flux_bottom = (export.data[-1] for export in model.exports)
    assert np.isclose(flux_top, -J_EXACT, rtol=1e-8)
    assert np.isclose(flux_bottom, J_EXACT, rtol=1e-8)

    interface = model.interfaces[0]
    lam = interface.multipliers[model.species[0]]
    V = lam.function_space
    n = V.dofmap.index_map.size_local
    x = V.tabulate_dof_coordinates()[:n]
    on_left = np.concatenate(MPI.COMM_WORLD.allgather(np.isclose(x[:, 0], 0.0)))
    values = owned_values(lam)
    assert on_left.sum() == 1  # the one end on the prescribed side
    assert np.isclose(values[on_left][0], 0.0)


def test_no_pinning_without_a_doubly_prescribed_end():
    model = build()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*fixed on both sides.*")
        model.initialise()


def test_shared_interface_id_is_rejected():
    model = build()
    bottom, top = model.interfaces[0].subdomains
    model.interfaces.append(F.Interface(5, (top, bottom), method="penalty"))
    with pytest.raises(ValueError, match="shares its id"):
        model.initialise()
