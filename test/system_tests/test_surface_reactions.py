import festim as F
import numpy as np

import ufl
import dolfinx.fem as fem


# TODO add this to the festim package
class FluxFromSurfaceReaction(F.SurfaceFlux):
    def __init__(self, reaction: F.SurfaceReactionBC):
        super().__init__(
            F.Species(),  # just a dummy species here
            reaction.subdomain,
        )
        self.reaction = reaction.flux_bcs[0]

    def compute(self, u, ds, entity_maps=None):
        self.value = fem.assemble_scalar(
            fem.form(self.reaction.value_fenics * ds(self.surface.id))
        )
        self.data.append(self.value)


def test_2_isotopes_no_pressure():
    """
    Runs a simple 1D hydrogen transport problem with 2 isotopes
    and 3 surface reactions on the right boundary.

    Then checks that the fluxes of the isotopes are consistent with the surface reactions
    by computing the flux of H and D (from the gradient) and comparing it to the fluxes
    computed from the surface reactions.

    Example: H + D <-> HD, H + H <-> HH, D + D <-> DD
    -D grad(c_H) n = 2*kr*c_H*c_H + kr*c_H*c_D

    Also checks the mass balance between the left and right boundary
    """
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 1000))
    my_mat = F.Material(name="mat", D_0=1, E_D=0)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    D = F.Species("D")
    my_model.species = [H, D]

    my_model.temperature = 500

    surface_reaction_hd = F.SurfaceReactionBC(
        reactant=[H, D],
        gas_pressure=0,
        k_r0=0.01,
        E_kr=0,
        k_d0=0,
        E_kd=0,
        subdomain=right,
    )

    surface_reaction_hh = F.SurfaceReactionBC(
        reactant=[H, H],
        gas_pressure=0,
        k_r0=0.02,
        E_kr=0,
        k_d0=0,
        E_kd=0,
        subdomain=right,
    )

    surface_reaction_dd = F.SurfaceReactionBC(
        reactant=[D, D],
        gas_pressure=0,
        k_r0=0.03,
        E_kr=0,
        k_d0=0,
        E_kd=0,
        subdomain=right,
    )

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=2, species=H),
        F.DirichletBC(subdomain=left, value=3, species=D),
        surface_reaction_hd,
        surface_reaction_hh,
        surface_reaction_dd,
    ]

    H_flux_right = F.SurfaceFlux(H, right)
    H_flux_left = F.SurfaceFlux(H, left)
    D_flux_right = F.SurfaceFlux(D, right)
    D_flux_left = F.SurfaceFlux(D, left)
    HD_flux = FluxFromSurfaceReaction(surface_reaction_hd)
    HH_flux = FluxFromSurfaceReaction(surface_reaction_hh)
    DD_flux = FluxFromSurfaceReaction(surface_reaction_dd)
    my_model.exports = [
        H_flux_left,
        H_flux_right,
        D_flux_left,
        D_flux_right,
        HD_flux,
        HH_flux,
        DD_flux,
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=5, transient=True)

    my_model.settings.stepsize = 0.1

    my_model.initialise()
    my_model.run()

    # TEST

    assert np.isclose(
        np.abs(H_flux_right.data[-1]),
        np.abs(H_flux_left.data[-1]),
        rtol=0.5e-2,
        atol=0.005,
    )
    assert np.isclose(
        np.abs(D_flux_right.data[-1]),
        np.abs(D_flux_left.data[-1]),
        rtol=0.5e-2,
        atol=0.005,
    )

    # check that H_flux_right == 2*HH_flux + HD_flux
    H_flux_from_gradient = -np.array(H_flux_right.data)
    H_flux_from_reac = 2 * np.array(HH_flux.data) + np.array(HD_flux.data)
    assert np.allclose(
        H_flux_from_gradient,
        H_flux_from_reac,
        rtol=0.5e-2,
        atol=0.005,
    )

    # check that D_flux_right == 2*DD_flux + HD_flux
    D_flux_from_gradient = -np.array(D_flux_right.data)
    D_flux_from_reac = 2 * np.array(DD_flux.data) + np.array(HD_flux.data)
    assert np.allclose(
        D_flux_from_gradient,
        D_flux_from_reac,
        rtol=0.5e-2,
        atol=0.005,
    )


def test_2_isotopes_with_pressure():
    """
    Runs a simple 1D hydrogen transport problem with 2 isotopes
    and 3 surface reactions on the right boundary.

    Then checks that the fluxes of the isotopes are consistent with the surface reactions
    by computing the flux of H and D (from the gradient) and comparing it to the fluxes
    computed from the surface reactions.

    Example: H + D <-> HD, H + H <-> HH, D + D <-> DD
    -D grad(c_H) n = 2*kr*c_H*c_H + kr*c_H*c_D - kd*P_H2

    Also checks the mass balance between the left and right boundary
    """
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 1000))
    my_mat = F.Material(name="mat", D_0=1, E_D=0)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    D = F.Species("D")
    my_model.species = [H, D]

    my_model.temperature = 500

    surface_reaction_hd = F.SurfaceReactionBC(
        reactant=[H, D],
        gas_pressure=0,
        k_r0=0.01,
        E_kr=0,
        k_d0=0,
        E_kd=0,
        subdomain=right,
    )

    surface_reaction_hh = F.SurfaceReactionBC(
        reactant=[H, H],
        gas_pressure=2,
        k_r0=0.02,
        E_kr=0,
        k_d0=0.1,
        E_kd=0,
        subdomain=right,
    )

    surface_reaction_dd = F.SurfaceReactionBC(
        reactant=[D, D],
        gas_pressure=0,
        k_r0=0.03,
        E_kr=0,
        k_d0=0,
        E_kd=0,
        subdomain=right,
    )

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=2, species=H),
        F.DirichletBC(subdomain=left, value=3, species=D),
        surface_reaction_hd,
        surface_reaction_hh,
        surface_reaction_dd,
    ]

    H_flux_right = F.SurfaceFlux(H, right)
    H_flux_left = F.SurfaceFlux(H, left)
    D_flux_right = F.SurfaceFlux(D, right)
    D_flux_left = F.SurfaceFlux(D, left)
    HD_flux = FluxFromSurfaceReaction(surface_reaction_hd)
    HH_flux = FluxFromSurfaceReaction(surface_reaction_hh)
    DD_flux = FluxFromSurfaceReaction(surface_reaction_dd)
    my_model.exports = [
        H_flux_left,
        H_flux_right,
        D_flux_left,
        D_flux_right,
        HD_flux,
        HH_flux,
        DD_flux,
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=5, transient=True)

    my_model.settings.stepsize = 0.1

    my_model.initialise()
    my_model.run()

    # TEST

    assert np.isclose(
        np.abs(H_flux_right.data[-1]),
        np.abs(H_flux_left.data[-1]),
        rtol=0.5e-2,
        atol=0.005,
    )
    assert np.isclose(
        np.abs(D_flux_right.data[-1]),
        np.abs(D_flux_left.data[-1]),
        rtol=0.5e-2,
        atol=0.005,
    )

    # check that H_flux_right == 2*HH_flux + HD_flux
    H_flux_from_gradient = -np.array(H_flux_right.data)
    H_flux_from_reac = 2 * np.array(HH_flux.data) + np.array(HD_flux.data)
    assert np.allclose(
        H_flux_from_gradient,
        H_flux_from_reac,
        rtol=0.5e-2,
        atol=0.005,
    )

    # check that D_flux_right == 2*DD_flux + HD_flux
    D_flux_from_gradient = -np.array(D_flux_right.data)
    D_flux_from_reac = 2 * np.array(DD_flux.data) + np.array(HD_flux.data)
    assert np.allclose(
        D_flux_from_gradient,
        D_flux_from_reac,
        rtol=0.5e-2,
        atol=0.005,
    )


def test_pressure_varies_in_time():
    """
    Runs a problem with a surface reaction and a time-dependent pressure
    on the right boundary.

    Then checks that the flux is consistent with the surface reaction
    """
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 1000))
    my_mat = F.Material(name="mat", D_0=1, E_D=0)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = 500

    t_pressure = 2
    pressure = 2
    k_d = 2

    surface_reaction_hh = F.SurfaceReactionBC(
        reactant=[H, H],
        gas_pressure=lambda t: ufl.conditional(ufl.gt(t, t_pressure), pressure, 0),
        k_r0=0,
        E_kr=0,
        k_d0=k_d,
        E_kd=0,
        subdomain=right,
    )

    my_model.boundary_conditions = [surface_reaction_hh]

    H_flux_right = F.SurfaceFlux(H, right)
    my_model.exports = [H_flux_right]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=5, transient=True)

    my_model.settings.stepsize = 0.1

    my_model.initialise()
    my_model.run()

    flux_as_array = np.array(H_flux_right.data)
    time_as_array = np.array(H_flux_right.t)

    expected_flux_before_pressure = 0
    computed_flux_before_pressure = flux_as_array[time_as_array <= t_pressure]
    assert np.allclose(computed_flux_before_pressure, expected_flux_before_pressure)

    expected_flux_after_pressure = -2 * k_d * pressure
    computed_flux_after_pressure = flux_as_array[time_as_array > t_pressure]
    print(computed_flux_after_pressure)
    print(expected_flux_after_pressure)
    assert np.allclose(
        computed_flux_after_pressure, expected_flux_after_pressure, rtol=1e-2
    )
