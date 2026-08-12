"""Deuterium TDS from a BaZr0.9Y0.1O2.95 (BZY) membrane -- FESTIM implementation of

    L. Yang, P.-C. A. Simon, W. Tang, M. Li, Z. Zhao, D. Ding, T. Fuerst,
    "Elucidating hydrogen isotope transport mechanisms in proton-conducting ceramics
    with trapping effects using TMAP8", Int. J. Hydrogen Energy 210 (2026) 153551,

reproducing the TMAP8 model of their Fig. 4 (compare with paper_figures/fig4e,f).
Parameters come from TMAP8's val-2g input files, which is the same model:
idaholab/TMAP8 test/tests/val-2g/{val-2g_trapping.i, parameters_*.params}.

Physics (all of it lives in FESTIM already):

  * three mobile species with their own Arrhenius diffusivity -- the hydroxyl OD.
    that carries the deuterium, oxygen vacancies V_O.. and electrons e'
  * one trap population, Eq. (2):
        dC_t/dt = alpha_t (chi N - C_t) C_OD / N - alpha_r C_t
  * two reversible kinetic surface reactions on both faces, Eqs. (6)-(9):
        D2O + V_O.. + O_O^x <-> 2 OD.        r_w = K1w P_D2O C_V C_Ox - K-1w C_OD^2
        D2  + 2 O_O^x       <-> 2 OD. + 2 e' r_d = K1d P_D2 C_Ox^2 - K-1d C_OD^2 C_e^2
    both are active in both environments; only the loading pressures differ, which is
    why the paper reports a D2 and a D2O flux for each environment
  * the TDS history of Fig. 2: 3600 s dissolution at 873 K, quench to 300 K, pump
    down, then a 0.5 K/s ramp in vacuum

Units: mol/m3, Pa, m, s, eV (festim.k_B); TMAP8 itself runs this case in at/um3
and um. Lattice oxygen O_O^x is not an unknown: the oxygen sublattice is shared,
C_Ox = 3 N - C_V - C_OD.

Run:  python pcc_bzy_tds.py     ->  pcc_bzy_dry.png, pcc_bzy_wet.png
"""

import matplotlib.pyplot as plt
import numpy as np
import ufl

import festim as F

N_A = 6.02214076e23  # 1/mol
R_GAS = 8.314  # J/mol/K, the unit of the tabulated molar entropies/enthalpies

# ---------------------------------------------------------------------------
# sample and lattice (supplementary Table 1)
# ---------------------------------------------------------------------------
L = 0.5e-3  # membrane thickness (m)
AREA = 7.7e-3 * 2.2e-3  # membrane area (m2), used to turn fluxes into mol/s
DENSITY = 5.98e6  # membrane density (g/m3)
M_BZY = 276.085  # g/mol

N_SITE = DENSITY / M_BZY  # formula units (mol/m3), 2.166e4
HYDRATION_LIMIT = 0.1
C_V0 = HYDRATION_LIMIT / 2 * N_SITE  # oxygen vacancies from the Y doping
# Lattice oxygen is not a constant: the three oxygen sites per formula unit are
# shared between O_O^x, the vacancies and the hydroxyls (val-2g_trapping.i).
N_O_SITES = 3 * N_SITE

# ---------------------------------------------------------------------------
# transport and surface kinetics
#
# "supp" = supplementary Table 1 (literature values, uncalibrated)
# "calibrated" = main-text Table 1 (the 18-parameter PSS calibration of Fig. 4e,f)
# ---------------------------------------------------------------------------
# The exponents are those of TMAP8's parameters_*.params files: the prefactors are
# tau_t0 = 4.8e{expo}, tau_r0 = 2.6e{expo}, chi = 3e{expo}, C_e0 = 10^{expo} * N,
# K1 = 2e{expo}, which is what turns the exponents into the paper's quoted values.
PARAMS = {
    "supp": dict(  # parameters_trapping_initial_validation.params
        D_0={"OD": 2e-9, "V_O": 1.021e-7, "e": 2.05e-2},
        E_D={"OD": 0.23, "V_O": 89216.77 / 96485, "e": 103818.22 / 96485},
        K1_wet=2 * 10**-33,  # m4/atom/Pa/s
        K1_dry=2 * 10**-41,
        dS_wet=-88.90,  # J/mol/K
        dH_wet=-79.5e3,  # J/mol
        dS_dry=-124.53,
        dH_dry=-79.5e3,
        C_e0_expo=-5.4,
        tau_t0=4.8 * 10**11,  # 1/s
        tau_r0=2.6 * 10**14,
        eps_t=0.38,  # eV
        eps_r=1.60,
        chi=3 * 10**-5,
    ),
    "calibrated": dict(  # parameters_trapping_calibrated_validation.params
        D_0={"OD": 1.90232846e-9, "V_O": 1.23735898e-7, "e": 2.06292148e-2},
        E_D={
            "OD": 1.21595541e-1,
            "V_O": 1.0034257e5 / 96485,
            "e": 9.53470966e4 / 96485,
        },
        K1_wet=2 * 10**-3.04026733e01,
        K1_dry=2 * 10**-4.4026596e01,
        dS_wet=-1.37384676e02,
        dH_wet=-1.56376367e05,
        dS_dry=-3.69926058e01,
        dH_dry=-1.12177032e05,
        C_e0_expo=-1.60701784,
        tau_t0=4.8 * 10**8.91574588,
        tau_r0=2.6 * 10**1.78981096e01,
        eps_t=4.67088776e-01,
        eps_r=1.24373578,
        chi=3 * 10**-2.55734086,
    ),
}
P = PARAMS["calibrated"]

C_E0 = 10 ** P["C_e0_expo"] * N_SITE  # initial electrons, a fraction of the lattice
# TMAP8 scales the tabulated diffusivity prefactor of the hydroxyl by sqrt(3/2)
D_0_OD = P["D_0"]["OD"] * np.sqrt(3 / 2)

# TMAP8 runs val-2g in atoms/um3 and um. The forward rates come out unit-independent
# once K1 [m4/atom/s] is multiplied by N_A, because both their terms are quadratic in
# concentration -- but the reverse D2 term is quartic (C_OD^2 C_e^2), so K_eq of
# Eq. (11), which carries concentration^2, needs the unit of the frame it was
# calibrated in.
DRY_KEQ_UNIT = (1e18 / N_A) ** 2  # (atoms/um3)^2 -> (mol/m3)^2

# ---------------------------------------------------------------------------
# TDS history (supplementary Table 1 + Fig. 2)
# ---------------------------------------------------------------------------
T_DISSOLVE, T_COOL_TAU, T_PUMP, T_RAMP = 3600.0, 600.0, 6200.0, 7200.0
T_HIGH, T_LOW, T_RATE = 873.0, 300.0, 0.5  # K, K, K/s
P_D2O_LOAD, P_D2_LOAD, P_VAC = 2.8e3, 1.33e3, 1e-5  # Pa
# P_VAC is the residual pressure after pumping; set it to 0 for perfect pumping.

# In TMAP8's val-2g the wet system leaves the D2 branch out of the hydroxyl flux
#   flux_on_OT_dry = 2 * flux_base_on_T2_dry + 2 * flux_base_on_T2O_dry
#   flux_on_OT_wet = 2 * flux_base_on_T2O_wet
# while still coupling that branch to the electrons and still reporting a D2 release.
# Deuterium is therefore not conserved in their wet system: the D2 channel drains the
# reported flux without draining the solid. Setting this True reproduces that
# convention; with it False (deuterium conserved) the fast D2 branch takes all the
# release and the D2O peak of Fig. 4(f) disappears.
TMAP8_WET_OT_FLUX_OMITS_D2 = True

# peak of the measured desorption flux, read off Fig. 4(e) and 4(f)
EXPERIMENT_PEAK = {  # environment -> {gas: (T in K, flux in mol/s)}
    "dry": {"D2": (1010, 1.48e-11), "D2O": (1100, 0.63e-11)},
    "wet": {"D2": (1000, 0.07e-9), "D2O": (1010, 1.93e-9)},
}
FINAL_TIME = T_RAMP + (1400.0 - T_LOW) / T_RATE  # stop where Fig. 4 stops


def temperature(t):
    """873 K plateau, exponential quench to 300 K, then a 0.5 K/s ramp."""
    if t < T_DISSOLVE:
        return T_HIGH
    if t < T_RAMP:
        return T_LOW + (T_HIGH - T_LOW) * np.exp(-(t - T_DISSOLVE) / T_COOL_TAU)
    return T_LOW + T_RATE * (t - T_RAMP)


def _pressure(t, loading):
    """Gas pressure as a ufl expression (t is a fem.Constant inside a BC).

    The gas that is not the loading gas is absent, not merely at the vacuum floor
    ("under dry environment ... the D2O pressure is null", Fig. 2) -- and it matters:
    K1_wet is large enough that 1e-5 Pa of D2O would keep hydrating the sample.
    """
    if loading == 0.0:
        return 0.0
    return ufl.conditional(ufl.lt(t, T_PUMP), loading, P_VAC)


def K_eq_wet(T):
    """Eq. (10). Dimensionless in concentration, so unit-independent (1/Pa)."""
    return ufl.exp(P["dS_wet"] / R_GAS - P["dH_wet"] / (R_GAS * T))


def K_eq_dry(T):
    """Eq. (11), converted to (mol/m3)^2/Pa."""
    return DRY_KEQ_UNIT * ufl.exp(P["dS_dry"] / R_GAS - P["dH_dry"] / (R_GAS * T))


def run(environment: str):
    """Run the full TDS cycle in "dry" (D2 loading) or "wet" (D2O loading)."""
    p_d2o = P_D2O_LOAD if environment == "wet" else 0.0
    p_d2 = P_D2_LOAD if environment == "dry" else 0.0

    # forward rates, converted from m4/atom/Pa/s to a molar rate in mol/m2/s
    k1_wet = P["K1_wet"] * N_A
    k1_dry = P["K1_dry"] * N_A

    def c_oxygen(c_OD, c_V):
        """Lattice oxygen left on the oxygen sublattice."""
        return N_O_SITES - c_V - c_OD

    def rate_wet(T, t, c_OD, c_V):
        """D2O + V_O.. + O_O^x <-> 2 OD.  (Eq. 8), positive = dissociation"""
        return k1_wet * (
            _pressure(t, p_d2o) * c_V * c_oxygen(c_OD, c_V) - c_OD**2 / K_eq_wet(T)
        )

    def rate_dry(T, t, c_OD, c_V, c_e):
        """D2 + 2 O_O^x <-> 2 OD. + 2 e'  (Eq. 9), positive = dissociation"""
        return k1_dry * (
            _pressure(t, p_d2) * c_oxygen(c_OD, c_V) ** 2
            - c_OD**2 * c_e**2 / K_eq_dry(T)
        )

    # --- mesh: 1200 cells, half of them in the 12 um layers at the two faces ---
    edge = np.linspace(0, 12e-6, 251)
    bulk = np.linspace(12e-6, L - 12e-6, 701)
    vertices = np.unique(np.concatenate([edge, bulk, L - edge[::-1]]).round(12))

    model = F.HydrogenTransportProblem()
    model.mesh = F.Mesh1D(vertices)

    bzy = F.Material(D_0=P["D_0"] | {"OD": D_0_OD}, E_D=P["E_D"], name="BZY")
    volume = F.VolumeSubdomain1D(id=1, borders=[0, L], material=bzy)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=L)
    model.subdomains = [volume, left, right]

    OD = F.Species("OD")  # hydroxyl: carries the deuterium
    V_O = F.Species("V_O")  # oxygen vacancies
    e = F.Species("e")  # electrons
    trapped = F.Species("trapped", mobile=False)
    empty = F.ImplicitSpecies(n=P["chi"] * N_SITE, others=[trapped], name="empty")
    model.species = [OD, V_O, e, trapped]

    model.temperature = temperature

    # Eq. (2): the 1/N of the paper's trapping term folds into the rate constant
    model.reactions = [
        F.ArrheniusReaction(
            reactant=[OD, empty],
            product=trapped,
            k_0=P["tau_t0"] / N_SITE,
            E_k=P["eps_t"],
            p_0=P["tau_r0"],
            E_p=P["eps_r"],
            volume=volume,
        )
    ]

    # --- surface kinetics -------------------------------------------------
    # F.SurfaceReactionBC only covers  k_r prod(C_i) - k_d P: one gas pressure, no
    # species on the dissociation side, same flux for every reactant. These reactions
    # need C_V/C_Ox/C_e in the rate and a different stoichiometric factor per species,
    # so they go in as species-dependent particle fluxes.
    omit_d2 = environment == "wet" and TMAP8_WET_OT_FLUX_OMITS_D2
    if omit_d2:

        def flux_on_od(T, t, c_OD, c_V, c_e):
            return 2 * rate_wet(T, t, c_OD, c_V)
    else:

        def flux_on_od(T, t, c_OD, c_V, c_e):
            return 2 * (rate_wet(T, t, c_OD, c_V) + rate_dry(T, t, c_OD, c_V, c_e))

    bcs = []
    for surface in (left, right):
        bcs += [
            F.ParticleFluxBC(  # 2 OD. produced by the surface reactions
                subdomain=surface,
                value=flux_on_od,
                species=OD,
                species_dependent_value={"c_OD": OD, "c_V": V_O, "c_e": e},
            ),
            F.ParticleFluxBC(  # one vacancy consumed per D2O dissociated
                subdomain=surface,
                value=lambda T, t, c_OD, c_V: -rate_wet(T, t, c_OD, c_V),
                species=V_O,
                species_dependent_value={"c_OD": OD, "c_V": V_O},
            ),
            F.ParticleFluxBC(  # 2 e' produced per D2 dissociated
                subdomain=surface,
                value=lambda T, t, c_OD, c_V, c_e: 2 * rate_dry(T, t, c_OD, c_V, c_e),
                species=e,
                species_dependent_value={"c_OD": OD, "c_V": V_O, "c_e": e},
            ),
        ]
    model.boundary_conditions = bcs

    model.initial_conditions = [
        F.InitialConcentration(value=0.0, volume=volume, species=OD),
        F.InitialConcentration(value=C_V0, volume=volume, species=V_O),
        F.InitialConcentration(value=C_E0, volume=volume, species=e),
        F.InitialConcentration(value=0.0, volume=volume, species=trapped),
    ]

    # --- exports -----------------------------------------------------------
    # D atoms leaving the surface, 2 per recombined molecule. TMAP8 reports the
    # left face only and scales the measurement by N_A/Area to compare with it.
    def flux_as_d2(**kw):
        return -2 * rate_dry(kw["T"], kw["t"], kw["OD"], kw["V_O"], kw["e"])

    def flux_as_d2o(**kw):
        return -2 * rate_wet(kw["T"], kw["t"], kw["OD"], kw["V_O"])

    quantities = {
        "d2": F.CustomQuantity(flux_as_d2, left, title="D2 left"),
        "d2o": F.CustomQuantity(flux_as_d2o, left, title="D2O left"),
        "mobile": F.TotalVolume(field=OD, volume=volume),
        "trapped": F.TotalVolume(field=trapped, volume=volume),
    }
    model.exports = list(quantities.values())

    model.settings = F.Settings(
        atol=1e-12,
        rtol=1e-8,
        max_iterations=30,
        final_time=FINAL_TIME,
        stepsize=F.Stepsize(
            initial_value=1e-2,
            growth_factor=1.15,
            cutback_factor=0.5,
            target_nb_iterations=4,
            max_stepsize=lambda t: 4.0 if t > T_RAMP else 50.0,
            milestones=[T_DISSOLVE, T_PUMP, T_RAMP, FINAL_TIME],
        ),
    )

    model.initialise()
    model.run()

    t = np.array(quantities["d2"].t)
    return {
        "t": t,
        "T": np.array([temperature(ti) for ti in t]),
        "d2": AREA * np.array(quantities["d2"].data),  # mol/s
        "d2o": AREA * np.array(quantities["d2o"].data),
        "mobile": np.array(quantities["mobile"].data),
        "trapped": np.array(quantities["trapped"].data),
    }


def plot(res, environment):
    """Desorption spectrum in the axes of the paper's Fig. 4, plus the inventory."""
    desorption = res["t"] > T_RAMP
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(
        res["T"][desorption],
        res["d2"][desorption],
        color="tab:blue",
        label="FESTIM: D from D$_2$",
    )
    ax[0].plot(
        res["T"][desorption],
        res["d2o"][desorption],
        color="tab:orange",
        label="FESTIM: D from D$_2$O",
    )
    for gas, colour in (("D2", "tab:blue"), ("D2O", "tab:orange")):
        T_exp, flux_exp = EXPERIMENT_PEAK[environment][gas]
        ax[0].plot(
            T_exp,
            flux_exp,
            "*",
            ms=13,
            color=colour,
            mec="k",
            mew=0.5,
            label=f"measured {gas} peak (Fig. 4)",
        )
    ax[0].set_xlim(300, 1400)
    ax[0].set_xlabel("Temperature (K)")
    ax[0].set_ylabel("Deuterium flux (mol/s)")
    ax[0].set_title(f"TDS spectrum, {environment} environment")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(res["t"], res["mobile"], label="mobile OD$^.$")
    ax[1].plot(res["t"], res["trapped"], label="trapped D")
    for milestone in (T_DISSOLVE, T_PUMP, T_RAMP):
        ax[1].axvline(milestone, color="0.8", lw=0.8, zorder=0)
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel("inventory (mol/m$^2$)")
    ax[1].set_title("Inventory")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"pcc_bzy_{environment}.png", dpi=130)

    peak = res["d2"][desorption].max()
    peak_T = res["T"][desorption][res["d2"][desorption].argmax()]
    peak_o = res["d2o"][desorption].max()
    peak_oT = res["T"][desorption][res["d2o"][desorption].argmax()]
    print(f"[{environment}] D2  peak {peak:.3e} mol/s at {peak_T:.0f} K")
    print(f"[{environment}] D2O peak {peak_o:.3e} mol/s at {peak_oT:.0f} K")
    print(
        f"[{environment}] released as D2  {np.trapezoid(res['d2'], res['t']):.3e} mol"
    )
    print(
        f"[{environment}] released as D2O {np.trapezoid(res['d2o'], res['t']):.3e} mol"
    )


if __name__ == "__main__":
    for environment in ("dry", "wet"):
        plot(run(environment), environment)
