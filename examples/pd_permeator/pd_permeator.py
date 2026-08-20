r"""Deuterium permeation through a self-supported Pd-25Ag tube.

After T. F. Fuerst, C. N. Taylor and M. Shimada, *Deuterium Permeation Through a
Self-Supported Palladium-Silver Membrane in Helium Gas Mixtures*, IEEE Trans. Plasma
Sci. **52** (2024) 3925, doi:10.1109/TPS.2024.3356857 -- the PreTEX experiment at INL.

A gas mixture of 3.95% D2 in He flows down the inside of a 152 mm Pd-25Ag tube and
deuterium permeates radially outward into a vacuum. The measured permeated flow rate
rises with the feed flow rate and then plateaus (their Figs. 4 and 5), and reproducing
that curve is the point of this example: the rise is *axial depletion*. At 100 sccm the
feed only carries 3.95 sccm of D2 and the tube extracts most of it, so permeation is
limited by what the gas stream delivers rather than by the membrane. A model that
prescribes the feed pressure cannot show this at all -- it gives a flat line at the
plateau value.

Getting the depletion means transporting the feed gas as its own field along the wall,
which is what a **codim-1 (manifold) subdomain** is for::

      y = L    :  retentate out (OutflowBC)                permeate: vacuum
               :
    feed gas   :                Pd-25Ag wall               |
    (1D        :  -> J ->    (2D, x across the wall)    -> J -> to the pumps
     manifold) :
      y = 0    :  3.95% D2 in He in (FixedConcentrationBC)

Four resistances act in series, all four from the paper:

1. gas-phase mass transfer from the bulk feed to the wall, eq. (2), through the
   coefficient K_T of the Sherwood correlation eq. (6);
2. dissociative adsorption on the feed surface, eq. (3);
3. interstitial diffusion across the wall, eq. (4);
4. recombinative desorption into the vacuum, eq. (5).

Resistances 1 and 2 are in series with nothing in between, so the surface pressure
P_S can be eliminated between eqs. (2) and (3) analytically::

    J = (k_d P_B - k_r c_w^2) / (1 + k_d R T / K_T)

which is a single flux depending on the manifold field (through P_B) and the wall
field, i.e. one ``ParticleFluxBC`` with ``species_dependent_value``. No extra unknown
is needed.

Units
-----
Everything is in **moles of D2**, as in the paper: k_d in mol_D2 m-2 s-1 Pa-1, k_r in
m4 mol_D2-1 s-1, and the wall concentration in mol_D2 m-3 (half the atomic
concentration). This is not a guess. Serra et al. (1998), where all the metal properties
originate, measure J by the pressure rise in a calibrated volume and report it in "moles
of gas", and define ``c = K_S p^0.5`` in the same basis; ``serra_consistency()`` checks
that the tabulated k_d, k_r and K_S close on each other under exactly the equations used
here. Working in that convention lets Table I be copied verbatim, and is why desorption
is a plain ``ParticleFluxBC`` rather than a ``festim.SurfaceReactionBC``, whose factor
of two assumes the atomic convention.

Geometry
--------
Codim-1 subdomains are cartesian-only, so the tube is unrolled into a plane. The wall
thickness used is the cylindrical diffusion resistance of eq. (4) referred to the inner
surface, r_i ln(r_o/r_i) = 74.2 um rather than the geometric 76.2 um, so that the
planar model reproduces the cylindrical flux per unit *inner* area exactly. Every area
in the script is then the inner one, 2 pi r_i L, which is also what the 2/r_i in the
gas-phase mass balance assumes. What is left over is the permeate-side recombination,
which really happens on an area 5% larger.

Note also that Table I of the paper lists r_i = 1.59 mm and r_o = 1.51 mm, which is the
two swapped: the stated 3.18 mm outer diameter and 76.2 um wall give r_i = 1.514 mm and
r_o = 1.590 mm, as used here.

Validation status
-----------------
*Against the measured curves* (``--compare``). The whole mixed-gas campaign, Fig. 4, has
been digitised by colour into ``fig4_digitised.csv``: 102 points over 16 conditions,
four temperatures by four pressures. Refitting k_d against all of them with the finite
element model, as their Section III-B does with theirs, gives 10.9% rms, against 36.0%
for their eq. (9) and 81.8% for Serra's constants. The fit returns

    k_d = 4.432e-5 exp(-16810/RT)

against their eq. (9) 7.99e-4 exp(-27600/RT) -- a factor 0.53 lower at 300 C and 0.33
at 450 C.

Sixteen conditions pin two parameters much harder than Fig. 5's two did (which fit to
5.3%), and the residual is structured rather than scattered. It grows steadily as the
temperature falls -- 2.4-5.2% across the 450 C conditions, 9.0-29.5% across the 300 C
ones -- and within each temperature it is worst at the lowest pressures. That is the
corner where the surface resistance dominates most, which is the same finger already
pointed at eqs. (2)-(5) below.

Part of the 300 C residual is not the model's fault: in that panel the measured 150 kPa
series sits *above* the 190 kPa one, which no monotonic model can reproduce and which is
not the ordering the other three temperatures show.

*How they seem to have fitted it* (``--per-condition``). Their Fig. 6 reports four k_d
values, one per temperature, each with an error bar -- and a fit giving one number per
temperature would have no spread to draw. Read as the scatter across the four pressures,
that says k_d was optimised condition by condition and only then collapsed onto an
Arrhenius law, which is also the only way the CM-O lines of their Fig. 4 can sit on
every pressure series at once, 300 C included. Running that procedure here:

* the fits become good, 3.1% rms on average and 8.8% at worst, against 10.9% for one
  global Arrhenius. So the temperature trend in the residual above is an artefact of
  forcing a single k_d(T), not a missing resistance;
* the spread across pressures at fixed temperature grows exactly as their error bars do,
  x1.08 at 450 C rising to x2.43 at 300 C against their x1.9 to x2.6 (theirs measured
  off the figure, and inflated by the marker and by whatever else they folded in);
* the 300 C 150 kPa anomaly is simply absorbed: that condition alone wants
  k_d = 2.3e-6, about twice its neighbours.

What it does **not** explain is the size of k_d. Collapsing the sixteen values onto an
Arrhenius law gives 3.522e-5 exp(-15545/RT), essentially where the global fit already
was and nowhere near their eq. (9). Nor is eq. (9) a misprint: digitising the markers of
Fig. 6 gives 8.0, 6.1, 3.3 and 2.6e-6 at 450 to 300 C, which eq. (9) reproduces to
6-14%. The factor of two to three is in the model, not in the estimator.

*Against their pure-D2 campaign* (``--uhp``). That one is reported in closed form, as
the permeability fit of eq. (7), so it needs no figure. Their optimised k_d reproduces
it to 9-11% at all four temperatures. Note that fitting k_d to *this* campaign gives
2.5e-4 exp(-23626/RT), about 1.8x the value the mixed-gas curves want -- the two
datasets do not quite agree on k_d through this model, which is the same tension in
another form.

*Against Serra et al. (1998)* (``--uhp``, first table). Every metal property comes from
there via Table I, and all of it checks out: ``sqrt(k_d/k_r)`` returns their K_S to
0.8%, ``D K_S`` returns the permeability they measured directly to 0.3%, Table I is
transcribed from them faithfully, and in the surface-limited limit :func:`flux_0d`
converges on their eq. [4], ``J = k_d p / 2``, to four digits. The convention is
molecular throughout, so there is no factor of two lurking between D2 and D atoms -- a
hypothesis this kills.

*The unexplained part.* Their Fig. 5 also draws the DM and CM lines, so those can be
compared model to model. The DM agrees closely (12.9 sccm here against ~13.0 read off
the figure, at 300 C and 90 kPa and 1000 sccm feed), which pins down the geometry, the
solubility, the diffusivity, the axial depletion and the sccm conversion. The CM does
not: 11.0 sccm here against ~6.3 in the figure. The two models differ only by the
surface terms, and those terms are now verified against Serra directly, so the
discrepancy is in how eqs. (2)-(5) are implemented on their side rather than in the
constants or in the convention. One candidate the papers do leave open: Serra obtained
k_1 by fitting J/p through their eq. [5], a series interpolation between the diffusion-
and surface-limited regimes, so a constant carried from that fit into a directly solved
resistance chain need not mean quite the same thing. That is a question for the authors,
not something to patch here.

*Against an independent solve* (``--compare``, last line). :func:`plug_flow` integrates
the same physics as a 1D gas balance with :func:`flux_0d` at each station -- the
structure of the paper's own 50-node code, sharing nothing with the finite element model
but the material properties. Over all 102 measured conditions they differ by -0.29% on
average and 2.67% at worst, the largest gaps being where extraction is strongest and
the Dirichlet inlet of :func:`solve` strains hardest. Refitting k_d through the reduced
model instead moves the answer by 2% in the prefactor and 0.4% in the activation
energy, and not at all in the rms.

*Internally.* The mass balance closes to better than 0.3% everywhere on the default
mesh, 0.7% at the one point where nearly all the feed is extracted, and the residual is
dominated by the *measurement* of the feed rather than by conservation (see
:func:`solve`). The permeated flow rate is mesh converged to four digits by 150 axial
and 4 radial cells.

Usage
-----
::

    python pd_permeator.py                      # one flow sweep, saves a figure
    mpirun -np 16 python pd_permeator.py --compare   # reproduce Fig. 4, refitting k_d
    python pd_permeator.py --uhp                # the pure-D2 calibration check
    python pd_permeator.py --temperature 723 --pressure 250e3
    python pd_permeator.py --kd serra --kd optimised   # compare CM and CM-O
    python pd_permeator.py --quick               # coarse mesh, three flow rates

Note that ``--d2-fraction 1.0`` does *not* reproduce their pure-D2 campaign. A regulated
pure gas does not deplete -- permeation lowers the velocity, not the partial pressure --
whereas the transported field here depletes as it is consumed. That case is ``--uhp``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np

import festim as F

# ----------------------------------------------------------------- constants

J_PER_MOL_TO_EV = 1.0 / 96485.33212  # activation energies are in J/mol in the paper

#: mol of gas per sccm per second, at 273.15 K and 101325 Pa
SCCM = 101325e-6 / (F.R * 273.15) / 60  # 7.4356e-7 mol/s

# --------------------------------------------------------------- the membrane

R_IN = 1.514e-3  # m, inner radius (feed side)
R_OUT = 1.590e-3  # m, outer radius (permeate side)
LENGTH = 0.152  # m

#: the planar stand-in for the cylindrical wall -- see the module docstring
WALL_THICKNESS = R_IN * np.log(R_OUT / R_IN)

PERMEATE_PRESSURE = 1.0  # Pa, the paper's assumed permeate-side D2 pressure

# Pd-25Ag, Serra et al. (1998) as reported in Table I. D2-molar basis, J/mol.
D_0, E_D = 1.87e-7, 24685.0  # m2/s
S_0, E_S = 0.184, -18531.0  # mol_D2 m-3 Pa-0.5; dissolution is exothermic

#: dissociation constants, mol_D2 m-2 s-1 Pa-1: Table I and eq. (9)
KD_LAWS = {
    "serra": (1.30e-2, 24780.0),
    "optimised": (7.99e-4, 27600.0),
}

# ------------------------------------------------------------------- the mesh

WALL_ID, GAS_ID = 1, 2  # cell tag / facet tag of the manifold
INLET_ID, OUTLET_ID, PERMEATE_ID = 3, 4, 5


@dataclass(frozen=True)
class Case:
    """One point of the experimental campaign."""

    temperature: float = 573.15  # K, membrane temperature
    total_pressure: float = 90e3  # Pa, feed side
    d2_fraction: float = 0.0395  # the premixed cylinder
    feed_sccm: float = 100.0  # total gas, not D2
    kd_law: str = "optimised"  # "serra" (CM) or "optimised" (CM-O)

    @property
    def d2_pressure(self) -> float:
        """The D2 partial pressure fed to the tube (Pa)."""
        return self.d2_fraction * self.total_pressure

    def __str__(self) -> str:
        return (
            f"{self.temperature - 273.15:.0f} C, "
            f"{self.total_pressure / 1e3:.0f} kPa, "
            f"{self.feed_sccm:.0f} sccm, k_d {self.kd_law}"
        )


# ------------------------------------------------------------------ properties


def diffusivity(temperature: float) -> float:
    """Interstitial D diffusivity in Pd-25Ag (m2/s)."""
    return D_0 * np.exp(-E_D / (F.R * temperature))


def sieverts_constant(temperature: float) -> float:
    """Solubility of D in Pd-25Ag (mol_D2 m-3 Pa-0.5)."""
    return S_0 * np.exp(-E_S / (F.R * temperature))


def dissociation_constant(case: Case) -> float:
    """k_d of eq. (3) (mol_D2 m-2 s-1 Pa-1)."""
    k_d0, e_kd = KD_LAWS[case.kd_law]
    return k_d0 * np.exp(-e_kd / (F.R * case.temperature))


def recombination_constant(case: Case) -> float:
    """k_r of eq. (5), from k_d through eq. (8) (m4 mol_D2-1 s-1).

    As in the paper, the Serra solubility is kept whichever k_d is used, so optimising
    k_d moves k_r with it.
    """
    return dissociation_constant(case) / sieverts_constant(case.temperature) ** 2


def measured_flux(case: Case) -> float:
    """The UHP D2 flux the experiment measured, from their eq. (7) (mol_D2 m-2 s-1).

    Their Arrhenius fit to the 90 kPa pure-D2 campaign, turned back into a flux through
    the Richardson equation it was derived with. This is the one piece of the paper's
    *data* available in closed form rather than only as a figure, which makes it the
    calibration anchor of :func:`uhp_calibration`.
    """
    permeability = 5.03e-8 * np.exp(-10700 / (F.R * case.temperature))
    return permeability * np.sqrt(case.d2_pressure) / WALL_THICKNESS


def binary_diffusivity(case: Case) -> float:
    """D2 in He (m2/s), from the Fuller correlation.

    The paper takes D_D2-He from Table I; the correlation is used here so that the
    example runs at any temperature and pressure. It gives 5.1e-4 m2/s at 300 C and
    90 kPa, and 1.5 cm2/s at 300 K and 1 atm, the textbook value for H2-He.
    """
    m_ab = 2 / (1 / 4.028 + 1 / 4.003)  # g/mol
    volumes = (6.12 ** (1 / 3) + 2.67 ** (1 / 3)) ** 2  # H2 and He diffusion volumes
    cm2_s = (
        1.43e-3
        * case.temperature**1.75
        / ((case.total_pressure / 1e5) * np.sqrt(m_ab) * volumes)
    )
    return cm2_s * 1e-4


def gas_velocity(case: Case) -> float:
    """Bulk axial velocity of the feed gas (m/s), at the membrane temperature.

    Uniform along the tube: even complete extraction of the D2 changes the total molar
    flow by less than the 3.95% the mixture contains. (This is *not* true of the
    pure-D2 campaign of their Fig. 3, where the flow drops as the gas permeates.)
    """
    volumetric = case.feed_sccm * SCCM * F.R * case.temperature / case.total_pressure
    return volumetric / (np.pi * R_IN**2)


def mass_transfer_coefficient(case: Case, *, entrance_only: bool = True) -> float:
    """K_T of eq. (2) (m/s), from the Sherwood correlation eq. (6).

    Their eq. (6) is the Leveque entrance-region correlation, and Re*Sc = d v / D
    turns it into a function of the velocity alone -- no viscosity needed. Note that at
    these conditions it returns Sh well below 3.66, the fully developed limit for a
    circular tube with a fixed wall concentration, which means it is being used outside
    its range of validity and underestimates K_T. Pass ``entrance_only=False`` to floor
    it at 3.66 and see how much that matters.
    """
    d = 2 * R_IN
    d_gas = binary_diffusivity(case)
    sherwood = 1.62 * (d**2 * gas_velocity(case) / (d_gas * LENGTH)) ** (1 / 3)
    if not entrance_only:
        sherwood = max(sherwood, 3.66)
    return sherwood * d_gas / d


# ---------------------------------------------------------------- the problem


def build(
    case: Case,
    n_axial: int = 200,
    n_radial: int = 6,
    entrance_only: bool = True,
    comm=MPI.COMM_WORLD,
):
    """Assembles the FESTIM problem for one operating point.

    Returns the problem and the three derived quantities that close its mass balance.

    ``comm`` is the communicator the mesh is built on, and so the one DOLFINx partitions
    it over. Leave it as ``COMM_WORLD`` for an ordinary parallel run of a single
    operating point. Pass ``MPI.COMM_SELF`` to give each rank a private serial problem
    instead, which is what :func:`map_solve` does to run many points at once -- FESTIM
    takes every communicator it needs from ``mesh.comm``, so the two modes need no other
    changes.
    """
    mesh = dolfinx.mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([WALL_THICKNESS, LENGTH])],
        [n_radial, n_axial],
    )

    wall = F.VolumeSubdomain(
        id=WALL_ID,
        material=F.Material(D_0=D_0, E_D=E_D * J_PER_MOL_TO_EV),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    # the feed gas: a 1D domain running along the inner face of the wall, with the real
    # binary diffusivity along it. FESTIM does not stabilise advection-diffusion, so the
    # cell Peclet number is worth watching (report() prints it), though this problem
    # turns out to be forgiving: the profile is a smooth decay, and the answer is
    # converged to four digits by 150 axial cells even at a cell Peclet of 10
    gas = F.VolumeSubdomain(
        id=GAS_ID,
        material=F.Material(D_0=binary_diffusivity(case), E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], 0.0),
    )
    # the two ends of that 1D domain, and the outer face of the wall
    inlet = F.SurfaceSubdomain(
        id=INLET_ID, dim=0, locator=lambda x: np.isclose(x[1], 0)
    )
    outlet = F.SurfaceSubdomain(
        id=OUTLET_ID, dim=0, locator=lambda x: np.isclose(x[1], LENGTH)
    )
    permeate = F.SurfaceSubdomain(
        id=PERMEATE_ID, locator=lambda x: np.isclose(x[0], WALL_THICKNESS)
    )

    c_wall = F.Species("c_wall", subdomains=[wall])  # mol_D2 / m3 of metal
    c_gas = F.Species("c_gas", subdomains=[gas])  # mol_D2 / m3 of gas

    k_d = dissociation_constant(case)
    k_r = recombination_constant(case)
    k_t = mass_transfer_coefficient(case, entrance_only=entrance_only)
    rt = F.R * case.temperature

    # eqs. (2) and (3) with the surface pressure eliminated. Positive into the wall.
    mass_transfer_penalty = 1 / (1 + k_d * rt / k_t)

    def feed_flux(c_g, c_w):
        return mass_transfer_penalty * (k_d * c_g * rt - k_r * c_w**2)

    boundary_conditions = [
        # the wall gains what dissociates on the feed side ...
        F.ParticleFluxBC(
            subdomain=gas,
            species=c_wall,
            value=feed_flux,
            species_dependent_value={"c_g": c_gas, "c_w": c_wall},
        ),
        # ... and loses what recombines into the vacuum, eq. (5)
        F.ParticleFluxBC(
            subdomain=permeate,
            species=c_wall,
            value=lambda c_w: k_d * PERMEATE_PRESSURE - k_r * c_w**2,
            species_dependent_value={"c_w": c_wall},
        ),
        F.FixedConcentrationBC(
            subdomain=inlet, value=case.d2_pressure / rt, species=c_gas
        ),
        # without this the outlet is a wall and the gas backs up against it
        F.OutflowBC(subdomain=outlet, species=c_gas),
    ]

    sources = [
        # the other half of the exchange: what the wall gains per unit membrane area,
        # the gas loses per unit gas volume. For a tube the two differ by the
        # perimeter-to-area ratio 2 pi r_i / (pi r_i^2)
        F.ParticleSource(
            volume=gas,
            species=c_gas,
            value=lambda c_g, c_w: -2 / R_IN * feed_flux(c_g, c_w),
            species_dependent_value={"c_g": c_gas, "c_w": c_wall},
        ),
    ]

    velocity = dolfinx.fem.Function(
        dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    )
    velocity.interpolate(
        lambda x: np.vstack(
            [np.zeros(x.shape[1]), np.full(x.shape[1], gas_velocity(case))]
        )
    )

    quantities = {
        "permeated": F.SurfaceFlux(field=c_wall, surface=permeate),
        "fed": F.SurfaceFlux(field=c_gas, surface=inlet),
        "retentate": F.SurfaceFlux(field=c_gas, surface=outlet),
    }

    problem = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[c_wall, c_gas],
        subdomains=[wall, gas, inlet, outlet, permeate],
        boundary_conditions=boundary_conditions,
        sources=sources,
        drift_terms=[F.AdvectionTerm(velocity=velocity, subdomain=gas, species=c_gas)],
        exports=list(quantities.values()),
        temperature=case.temperature,
        settings=F.Settings(atol=1e-8, rtol=1e-10, transient=False),
    )
    problem.show_progress_bar = False
    return problem, (wall, c_wall), (gas, c_gas), quantities


def seed_newton(case: Case, *guesses) -> None:
    """Starts Newton from the Sieverts equilibrium with the feed rather than from zero.

    The desorption terms go as ``-k_r c^2``, whose derivative vanishes at ``c = 0``:
    from an all-zero guess the first Newton step has nothing holding it back, overshoots
    by ten orders of magnitude, and needs ~50 iterations to crawl back. Starting
    anywhere sensible costs nothing and converges in a handful.

    A steady problem has no initial condition to carry the guess (FESTIM raises if you
    give it one), so it is written straight into the solution vector.
    """
    for (subdomain, species), value in guesses:
        index = species.subdomain_to_index[subdomain]
        subdomain.u.sub(index).interpolate(
            lambda x, value=value: np.full(x.shape[1], value)
        )


def solve(case: Case, **kwargs) -> dict:
    """Runs one operating point and reports the flow rates in sccm.

    Mind which area each flux is per. A quantity on a wall facet is per unit membrane
    area, so unrolling it back onto the tube means multiplying by the inner
    circumference. A quantity at the end of the *manifold* is per unit area of the thing
    the manifold's own equation is written per unit volume of -- the gas -- so it
    multiplies by the flow cross-section instead. Signs are FESTIM's: positive leaves
    the domain the species lives in.

    ``fed`` is the D2 the model actually receives, and it exceeds the nominal
    ``feed_sccm * d2_fraction`` by a few percent. That is the Dirichlet inlet: the
    concentration falls downstream, so diffusion carries D2 *forwards* across the inlet
    and adds to the advective supply. The physically right condition is Danckwerts',
    which fixes the total flux rather than the concentration, but the boundary of a
    manifold only accepts a ``FixedConcentrationBC``. The discrepancy is largest where
    extraction is strongest -- 3% at 100 sccm, under 1% at 1000 sccm -- and extraction
    is reported against ``fed`` rather than the nominal value so it stays below 100%.
    """
    problem, wall, gas, quantities = build(case, **kwargs)
    problem.initialise()
    seed_newton(
        case,
        (wall, sieverts_constant(case.temperature) * np.sqrt(case.d2_pressure)),
        (gas, case.d2_pressure / (F.R * case.temperature)),
    )
    problem.run()

    per_membrane_area = 2 * np.pi * R_IN
    per_flow_area = np.pi * R_IN**2
    sccm = {
        "permeated": quantities["permeated"].data[-1] * per_membrane_area / SCCM,
        "fed": -quantities["fed"].data[-1] * per_flow_area / SCCM,
        "retentate": quantities["retentate"].data[-1] * per_flow_area / SCCM,
    }
    # what comes in must go out. The residual is dominated not by the conservation of
    # the scheme but by the measurement of ``fed``: a flux at a codim-2 point is the
    # pointwise gradient of a P1 field, which is only first-order accurate
    sccm["balance"] = (sccm["fed"] - sccm["retentate"] - sccm["permeated"]) / sccm[
        "fed"
    ]
    sccm["extraction"] = sccm["permeated"] / sccm["fed"]
    return sccm


# ------------------------------------------------ the same physics, without the tube


def flux_0d(case: Case, *, with_mass_transfer: bool = True) -> float:
    """The same resistance chain at a single point, solved directly (mol_D2 m-2 s-1).

    No axial transport, no depletion: the feed sits at its inlet partial pressure the
    whole way. Two uses.

    It is an *independent* implementation of eqs. (2)-(5) -- a scalar root find rather
    than a finite element problem -- so agreement with :func:`solve` in the limit of
    negligible extraction checks the manifold coupling, the 2/r_i conversion and the
    export areas all at once. And because pure D2 at a regulated pressure does not
    deplete (permeating gas lowers the velocity, not the partial pressure), this is the
    right model for their UHP campaign, which :func:`uhp_calibration` compares against.
    """
    from scipy.optimize import brentq

    k_d = dissociation_constant(case)
    k_r = recombination_constant(case)
    diffusion = diffusivity(case.temperature)
    penalty = (
        1 / (1 + k_d * F.R * case.temperature / mass_transfer_coefficient(case))
        if with_mass_transfer
        else 1.0
    )

    def residual(flux):
        # eq. (3) with the gas-phase resistance folded in, inverted for the feed-side
        # concentration; eq. (5) inverted for the permeate-side one; eq. (4) must then
        # carry exactly the flux we assumed
        feed_squared = (k_d * case.d2_pressure - flux / penalty) / k_r
        if feed_squared < 0:
            return -np.inf
        permeate = np.sqrt(max((flux + k_d * PERMEATE_PRESSURE) / k_r, 0.0))
        return diffusion * (np.sqrt(feed_squared) - permeate) / WALL_THICKNESS - flux

    ceiling = k_d * case.d2_pressure * penalty
    return brentq(residual, 0.0, ceiling * (1 - 1e-12), xtol=1e-20, rtol=1e-14)


def plug_flow(case: Case) -> float:
    """The paper's own model structure: a 1D gas balance with :func:`flux_0d` per node.

    Their MATLAB code segments the permeator into 50 axial nodes and solves eqs. (2)-(5)
    at each, with the D2 carried between them by advection alone. Integrating

    .. math::

        \\frac{\\mathrm{d}\\dot n_{D_2}}{\\mathrm{d}z} = -2\\pi r_i J(P_{D_2}(z))

    is the continuous version of that, and it is fast enough to sit inside an optimiser,
    which the finite element model is not -- :func:`fit_dissociation_constant` uses it.

    It is also the check that matters most. The two share no code beyond the material
    properties: one is a stiff ODE in the gas flow, the other a coupled finite element
    problem with the gas on a manifold. They agree to 0.3% wherever the Dirichlet inlet
    of :func:`solve` is not straining (see its docstring), which says the manifold
    machinery is solving the problem the paper poses.
    """
    from scipy.integrate import solve_ivp

    total_in = case.feed_sccm * SCCM  # mol/s of gas, D2 and carrier together
    d2_in = total_in * case.d2_fraction
    perimeter = 2 * np.pi * R_IN

    def shed(_, state):
        d2_flow = state[0]
        if d2_flow <= 0:
            return [0.0]
        # the carrier stays, so the partial pressure falls with the D2 flow
        total_flow = total_in - (d2_in - d2_flow)
        local = Case(
            **{
                **case.__dict__,
                "d2_fraction": d2_flow / total_flow,
            }
        )
        return [-perimeter * flux_0d(local)]

    solution = solve_ivp(
        shed, (0.0, LENGTH), [d2_in], rtol=1e-9, atol=1e-18, max_step=LENGTH / 400
    )
    return (d2_in - solution.y[0, -1]) / SCCM


def serra_consistency(temperature: float = 573.15) -> None:
    """Checks the surface chain against its primary source, Serra et al. (1998).

    Everything the model uses for the metal comes from that paper through Table I of
    Fuerst et al., so it is worth confirming that the constants mean here what they
    meant there. Three things, all of which have to hold at once:

    * Serra measures J by the pressure rise in a calibrated volume and states it in
      "moles of gas", and defines ``c = K_S p^0.5`` in the same basis, so the whole
      chain is per mole of **D2**, not of D atoms. That is the convention used here.
    * their K_S is not independent: ``k_2 = k_1 / K_S^2``, which is eq. (8) of Fuerst.
      So ``sqrt(k_d/k_r)`` must return the tabulated K_S, and ``D K_S`` must return the
      permeability they measured directly.
    * in the surface-limited limit -- diffusion fast enough to flatten the profile --
      their eq. [4] gives ``J = k_d p / 2``, the factor of two being the two surfaces in
      series. :func:`flux_0d` has to reproduce that as D grows.
    """
    global D_0  # restored at the end

    gas_constant_t = F.R * temperature
    k_d = 1.30e-2 * np.exp(-24780 / gas_constant_t)  # Serra eq. [9], = Fuerst's k_d
    k_r = 0.390 * np.exp(-61842 / gas_constant_t)  # Serra eq. [9], = Fuerst's k_r

    print(f"\nSerra et al. (1998), D2 at {temperature - 273.15:.0f} C")
    print(f"  sqrt(k_d/k_r)  = {np.sqrt(k_d / k_r):8.4f}")
    print(f"  K_S as reported= {sieverts_constant(temperature):8.4f}  (their eq. [7])")
    print(f"  D * K_S        = {D_0 * S_0:.3e} exp(-6154/RT)")
    print("  permeability   = 3.430e-08 exp(-6156/RT)  (their eq. [7], measured)")

    surface_limited = k_d * 90e3 / 2  # their eq. [4]
    case = Case(
        temperature=temperature, total_pressure=90e3, d2_fraction=1.0, kd_law="serra"
    )
    original = D_0
    print(f"  surface-limited flux k_d p / 2 = {surface_limited:.5f} mol_D2 m-2 s-1")
    for scale in (1e2, 1e4, 1e6):
        D_0 = original * scale  # drive the bulk resistance to nothing
        flux = flux_0d(case, with_mass_transfer=False)
        print(f"    D x {scale:.0e}: {flux:9.5f}   ratio {flux / surface_limited:.4f}")
    D_0 = original


def uhp_calibration(temperatures=(573.15, 623.15, 673.15, 723.15)) -> None:
    """Checks the resistance chain against the one piece of the data in closed form.

    Their first campaign fed pure D2 at 90 kPa and reported the result as the
    permeability fit of eq. (7). Because there is no carrier gas there is no boundary
    layer and no axial depletion, so the whole of eqs. (3)-(5) can be tested against it
    without touching a figure. It is also the cleanest place to see what the k_d
    optimisation of their Section III-B is doing.
    """
    print("\nUHP D2 at 90 kPa: their eq. (7) against the model  [mol_D2 m-2 s-1]")
    print(
        f"  {'T (C)':>7} {'measured':>10} {'DM':>10} {'CM':>10} {'CM-O':>10}"
        f" {'CM-O/meas':>10}"
    )
    for temperature in temperatures:
        case = Case(temperature=temperature, total_pressure=90e3, d2_fraction=1.0)
        measured = measured_flux(case)
        # the diffusion-limited model, eq. (1): no surface resistance at all
        diffusion_limited = (
            diffusivity(temperature)
            * sieverts_constant(temperature)
            * np.sqrt(case.d2_pressure)
            / WALL_THICKNESS
        )
        complete = flux_0d(
            Case(**{**case.__dict__, "kd_law": "serra"}), with_mass_transfer=False
        )
        optimised = flux_0d(
            Case(**{**case.__dict__, "kd_law": "optimised"}), with_mass_transfer=False
        )
        print(
            f"  {temperature - 273.15:7.0f} {measured:10.4f} {diffusion_limited:10.4f}"
            f" {complete:10.4f} {optimised:10.4f} {optimised / measured:10.2f}"
        )


# ------------------------------------------------- comparison with the measured curves

MEASUREMENTS = "fig4_digitised.csv"


def load_measurements(filename: str = MEASUREMENTS) -> dict:
    """The digitised campaign, grouped by (temperature, pressure).

    Defaults to Fig. 4 -- 16 conditions, 102 points -- rather than Fig. 5's two. Both
    files have the same columns, so ``fig5_digitised.csv`` can be passed instead.

    Prefer Fig. 4 where they overlap. The two figures plot the same 300 C / 90 kPa and
    450 C / 250 kPa data, and their ordinates agree to about 0.1 sccm, but their
    abscissae do not: Fig. 4 puts every point on the 100 sccm steps the campaign
    actually used, to within 3.4 sccm, while Fig. 5 spaces its two series differently
    from each other and neither lands on the round numbers.
    """
    import csv

    path = Path(__file__).parent / filename
    with path.open() as stream:
        lines = [line for line in stream if not line.startswith("#")]

    series: dict = {}
    for row in csv.DictReader(lines):
        key = (float(row["temperature_K"]), float(row["total_pressure_Pa"]))
        entry = series.setdefault(
            key,
            {"temperature": key[0], "pressure": key[1], "feed": [], "permeated": []},
        )
        entry["feed"].append(float(row["feed_sccm"]))
        entry["permeated"].append(float(row["permeated_sccm"]))
    return series


def _case_for(entry: dict, feed: float, law: str) -> Case:
    return Case(
        temperature=entry["temperature"],
        total_pressure=entry["pressure"],
        feed_sccm=feed,
        kd_law=law,
    )


def map_plug_flow(cases: list[Case]) -> np.ndarray:
    """:func:`plug_flow` over many operating points, spread across the MPI ranks.

    What makes ``--compare`` slow is not one big solve, it is a few thousand small
    independent ones: the optimiser walks 102 operating points per residual evaluation
    and needs tens of those. That parallelises over **points**, not over a mesh, so this
    deals the cases round-robin across ``COMM_WORLD`` and allgathers the answers. Every
    rank comes out holding the whole vector, so the optimiser downstream takes the same
    steps everywhere and no rank has to lead. Run it with::

        mpirun -np 16 python pd_permeator.py --compare

    The finite element path is a different kind of parallel and must not be mixed in
    under one ``mpirun``: :func:`solve` hands ``COMM_WORLD`` to DOLFINx, which uses
    it to partition the *mesh*. Both are legitimate; they just cannot own the same
    communicator at once. Nothing here calls :func:`solve`, which is what makes this
    safe.

    Every rank must reach this call. That is why :func:`compare` runs everywhere and
    guards only its printing, rather than being wrapped in a rank-0 test the way the
    cheap serial reports are.
    """
    comm = MPI.COMM_WORLD
    mine = [
        (index, plug_flow(case))
        for index, case in enumerate(cases)
        if index % comm.size == comm.rank
    ]
    values = np.empty(len(cases))
    for chunk in comm.allgather(mine):
        for index, value in chunk:
            values[index] = value
    return values


#: mesh for the swept runs. Converged to 0.3% of an 800x12 mesh at the most strongly
#: extracting point, and a fifth of a second per solve, which is what makes it usable
#: inside the fit.
SWEEP_MESH = {"n_axial": 150, "n_radial": 4}


def map_solve(cases: list[Case], **mesh_kwargs) -> np.ndarray:
    """:func:`solve` over many operating points, spread across the MPI ranks.

    The same deal as :func:`map_plug_flow`, but running the actual finite element
    model. Each rank builds its mesh on ``MPI.COMM_SELF``, so instead of one problem
    partitioned over all the ranks there are as many independent serial problems as
    there are ranks -- the right decomposition when the work is hundreds of small
    unrelated solves rather than one large one. FESTIM takes its communicators from
    ``mesh.comm``, so nothing else has to know.

    Note that this is not more expensive than the reduced model: on the mesh of
    :data:`SWEEP_MESH` a solve takes about 0.2 s, against 0.14 s for :func:`plug_flow`.

    .. warning::

        **Pin the thread count.** Each rank here is a whole serial FESTIM problem, and
        its linear algebra will happily open a thread per core -- 34 of them per rank in
        one measurement, so 16 ranks put 544 threads on 16 cores and everything crawls.
        The ranks *are* the parallelism; there is nothing left for threads to do::

            mpirun -np 16 -x OMP_NUM_THREADS=1 python pd_permeator.py --compare

        This does not arise in an ordinary FESTIM run, where one problem is partitioned
        across the ranks and each holds a fraction of the mesh.
    """
    comm = MPI.COMM_WORLD
    settings = {**SWEEP_MESH, **mesh_kwargs, "comm": MPI.COMM_SELF}
    mine = [
        (index, solve(case, **settings)["permeated"])
        for index, case in enumerate(cases)
        if index % comm.size == comm.rank
    ]
    values = np.empty(len(cases))
    for chunk in comm.allgather(mine):
        for index, value in chunk:
            values[index] = value
    return values


def _flatten(measurements: dict, law: str) -> tuple[list[Case], np.ndarray]:
    """Every digitised point as a case to run and the value it should reproduce."""
    cases, measured = [], []
    for entry in measurements.values():
        for feed, permeated in zip(entry["feed"], entry["permeated"], strict=True):
            cases.append(_case_for(entry, feed, law))
            measured.append(permeated)
    return cases, np.array(measured)


def rms_error(law: str, measurements: dict | None = None, model=None) -> float:
    """Fractional rms disagreement of one k_d law with every digitised point."""
    measurements = measurements or load_measurements()
    cases, measured = _flatten(measurements, law)
    return float(
        np.sqrt(np.mean(np.square(np.log((model or map_solve)(cases) / measured))))
    )


def fit_dissociation_constant(
    measurements: dict | None = None, model=None
) -> tuple[float, float]:
    """Optimises k_d against the measured curves, as their Section III-B does.

    Two parameters, a pre-exponential and an activation energy, against every digitised
    point at once. :func:`plug_flow` stands in for the finite element model here: it is
    the same physics and agrees with it to a fraction of a percent, but it is fast
    enough to sit in the residual loop, and through :func:`map_plug_flow` it spreads
    over the ranks.
    """
    from scipy.optimize import least_squares

    measurements = measurements or load_measurements()
    model = model or map_solve
    _, measured = _flatten(measurements, "optimised")

    def residuals(parameters):
        KD_LAWS["_fit"] = (np.exp(parameters[0]), parameters[1])
        cases, _ = _flatten(measurements, "_fit")
        return np.log(model(cases) / measured)

    guess = [np.log(KD_LAWS["optimised"][0]), KD_LAWS["optimised"][1]]
    fit = least_squares(residuals, guess, x_scale=[1.0, 1e4], diff_step=1e-3)
    KD_LAWS.pop("_fit", None)
    return float(np.exp(fit.x[0])), float(fit.x[1])


def point_model(reduced: bool = False):
    """One operating point in, sccm out. The finite element model, or the reduced one.

    The two agree to -0.29% on average over the measured grid (``--compare`` prints
    that), so for a scan whose conclusion is about the *fitting procedure* rather than
    about the transport, the reduced model is a sound and much cheaper stand-in.
    """
    if reduced:
        return plug_flow
    return lambda case: solve(case, **SWEEP_MESH, comm=MPI.COMM_SELF)["permeated"]


def fit_per_condition(measurements: dict | None = None, reduced: bool = False) -> dict:
    """Optimise k_d separately for each (T, P), rather than one Arrhenius for all.

    Their Fig. 6 reports four k_d values, one per temperature, **each with an error
    bar**. A fit that produced a single number per temperature would have no spread to
    draw, so the bars are most naturally read as the scatter across the four pressures
    -- meaning k_d was optimised condition by condition and only then collapsed onto an
    Arrhenius law. That would explain why the CM-O lines of their Fig. 4 sit on every
    pressure series individually, including the 300 C panel where the measured 150 kPa
    curve runs *above* the 190 kPa one. No single k_d(T) can reproduce that inversion;
    one k_d per condition can, trivially.

    This runs that procedure: sixteen one-parameter fits. Parallelism is over conditions
    rather than over points -- each rank takes whole conditions and solves them with
    private ``COMM_SELF`` meshes, so there is no collective call inside the optimiser
    and the ranks never wait on each other.

    Returns ``{(temperature, pressure): (k_d, rms)}``.
    """
    from scipy.optimize import least_squares

    measurements = measurements or load_measurements()
    evaluate = point_model(reduced)
    comm = MPI.COMM_WORLD
    items = sorted(measurements.items())
    key = f"_one_{comm.rank}"

    mine = {}
    for index, ((temperature, pressure), entry) in enumerate(items):
        if index % comm.size != comm.rank:
            continue
        measured = np.array(entry["permeated"])

        def residuals(parameters, entry=entry, measured=measured):
            # a fixed value rather than an Arrhenius: zero activation energy
            KD_LAWS[key] = (np.exp(parameters[0]), 0.0)
            modelled = [evaluate(_case_for(entry, feed, key)) for feed in entry["feed"]]
            return np.log(np.array(modelled) / measured)

        start = np.log(dissociation_constant(_case_for(entry, 100.0, "optimised")))
        fit = least_squares(residuals, [start], diff_step=1e-3)
        mine[temperature, pressure] = (
            float(np.exp(fit.x[0])),
            float(np.sqrt(np.mean(np.square(fit.fun)))),
        )
    KD_LAWS.pop(key, None)

    fitted = {}
    for chunk in comm.allgather(mine):
        fitted.update(chunk)
    return fitted


def report_per_condition_fit(fitted: dict) -> tuple[float, float]:
    """Prints the per-condition k_d values and Arrhenius-fits them, as Fig. 6 does.

    Returns the pre-exponential and activation energy of that fit, which is the number
    to hold against their eq. (9) -- it is produced by the same estimator.
    """
    temperatures = sorted({t for t, _ in fitted}, reverse=True)
    pressures = sorted({p for _, p in fitted})

    print("\nk_d fitted to each condition on its own  [mol_D2 m-2 s-1 Pa-1]")
    header = "  T (C)  " + "".join(f"{p / 1e3:>11.0f} kPa" for p in pressures)
    print(header + "     spread   their eq.(9)")
    for temperature in temperatures:
        row = [fitted.get((temperature, p)) for p in pressures]
        values = [entry[0] for entry in row if entry]
        spread = max(values) / min(values)
        paper = dissociation_constant(Case(temperature=temperature, kd_law="optimised"))
        cells = "".join(f"{v[0]:>15.3e}" if v else f"{'--':>15}" for v in row)
        print(f"  {temperature - 273.15:5.0f}{cells}   x{spread:5.2f}   {paper:11.3e}")

    residuals = [entry[1] for entry in fitted.values()]
    print(
        f"\n  rms per condition with its own k_d: mean {100 * np.mean(residuals):.1f}%,"
        f" worst {100 * max(residuals):.1f}%"
    )

    # collapse to an Arrhenius law the way Fig. 6 does
    inverse_t = np.array([1.0 / (F.R * t) for t, _ in fitted])
    log_kd = np.array([np.log(value) for value, _ in fitted.values()])
    slope, intercept = np.polyfit(inverse_t, log_kd, 1)
    prefactor, activation = float(np.exp(intercept)), float(-slope)
    print(
        f"  Arrhenius through them: {prefactor:.3e} exp(-{activation:.0f}/RT)"
        f"   [their eq. (9): {KD_LAWS['optimised'][0]:.2e}"
        f" exp(-{KD_LAWS['optimised'][1]:.0f}/RT)]"
    )
    return prefactor, activation


def per_condition_error(law: str, measurements: dict, model=None) -> list[tuple]:
    """Fractional rms disagreement condition by condition, worst first.

    One collective call for the whole grid rather than one per condition, so the ranks
    stay busy.
    """
    cases, measured = _flatten(measurements, law)
    residuals = np.log((model or map_solve)(cases) / measured)

    rows, start = [], 0
    for (temperature, pressure), entry in measurements.items():
        stop = start + len(entry["feed"])
        rows.append(
            (
                float(np.sqrt(np.mean(np.square(residuals[start:stop])))),
                temperature,
                pressure,
                stop - start,
            )
        )
        start = stop
    return sorted(rows, reverse=True)


def compare(
    refit: bool = True,
    filename: str = "pd_permeator_fig4.png",
    reduced: bool = False,
) -> None:
    """The model against the digitised campaign, optionally refitting k_d first.

    Reproduces Fig. 4: one panel per temperature, one curve per total pressure.

    Everything here -- the fit, the errors and the curves that get drawn -- runs the
    **finite element model** through :func:`map_solve`, which is the whole point of the
    example. Pass ``reduced=True`` to put :func:`plug_flow` in its place instead. That
    is worth doing once: the two share no code beyond the material properties, so
    comparing the two sets of numbers is a check on the manifold coupling across all
    102 points rather than at the handful :func:`flux_0d` covers.
    """
    rank = MPI.COMM_WORLD.rank
    model = map_plug_flow if reduced else map_solve
    measurements = load_measurements()
    count = sum(len(entry["feed"]) for entry in measurements.values())
    laws = ["serra", "optimised"]
    if rank == 0:
        print(f"\nmodel: {'reduced (plug flow)' if reduced else 'FESTIM'}")
    if refit:
        KD_LAWS["fitted"] = fit_dissociation_constant(measurements, model=model)
        laws.append("fitted")
        if rank == 0:
            print(
                f"k_d refitted to {count} points: {KD_LAWS['fitted'][0]:.3e}"
                f" exp(-{KD_LAWS['fitted'][1]:.0f}/RT)"
                f"   [their eq. (9): {KD_LAWS['optimised'][0]:.2e}"
                f" exp(-{KD_LAWS['optimised'][1]:.0f}/RT)]"
            )

    errors = {law: rms_error(law, measurements, model=model) for law in laws}
    by_condition = per_condition_error(laws[-1], measurements, model=model)
    if rank == 0:
        print(f"\nrms disagreement with the {count} digitised points")
        for law in laws:
            print(f"  k_d {law:10s} {100 * errors[law]:5.1f}%")
        print(f"\nby condition, k_d {laws[-1]}, worst first")
        for error, temperature, pressure, n in by_condition:
            print(
                f"  {temperature - 273.15:3.0f} C  {pressure / 1e3:3.0f} kPa"
                f"  ({n:2d} pts)  {100 * error:5.1f}%"
            )

    # how far the finite element model and the reduced one drift apart over the whole
    # grid, rather than at the few points flux_0d covers
    if not reduced:
        cases, _ = _flatten(measurements, laws[-1])
        drift = map_solve(cases) / map_plug_flow(cases) - 1
        if rank == 0:
            print(
                f"\nFESTIM against the reduced model over all {count} points: mean"
                f" {100 * drift.mean():+.2f}%, worst {100 * np.abs(drift).max():.2f}%"
            )

    # the model curves, all conditions and laws in one collective call
    temperatures = sorted({t for t, _ in measurements}, reverse=True)
    pressures = sorted({p for _, p in measurements})
    feeds = np.linspace(50, 1050, 21)
    curve_cases, curve_keys = [], []
    for key, entry in measurements.items():
        for law in laws:
            curve_keys.append((key, law))
            curve_cases.extend(_case_for(entry, feed, law) for feed in feeds)
    curves = model(curve_cases).reshape(len(curve_keys), len(feeds))
    curves = dict(zip(curve_keys, curves, strict=True))

    if rank != 0:
        return

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    colours = dict(zip(pressures, ["C2", "C1", "C7", "C0"], strict=False))
    styles = {"serra": ":", "optimised": "--", "fitted": "-"}

    figure, panels = plt.subplots(
        len(temperatures), 1, figsize=(6.5, 9), sharex=True, layout="constrained"
    )
    for axes, temperature in zip(panels, temperatures, strict=True):
        for pressure in pressures:
            entry = measurements.get((temperature, pressure))
            if entry is None:
                continue
            colour = colours[pressure]
            axes.plot(
                entry["feed"],
                entry["permeated"],
                "o",
                ms=4,
                color=colour,
                label=f"{pressure / 1e3:.0f} kPa",
            )
            for law in laws:
                axes.plot(
                    feeds,
                    curves[(temperature, pressure), law],
                    styles[law],
                    color=colour,
                    lw=1.0,
                )
        axes.set_ylabel("permeated (sccm)")
        axes.set_ylim(0, 12)
        axes.set_xlim(0, 1100)
        axes.annotate(
            f"T = {temperature - 273.15:.0f} $\\degree$C",
            (0.97, 0.08),
            xycoords="axes fraction",
            ha="right",
        )
    panels[0].legend(fontsize=7, ncols=4, loc="upper left")
    panels[0].set_title(
        "Fig. 4 of Fuerst et al. (2024), digitised, against this model\n"
        "lines: $k_d$ " + ", ".join(f"{law} ({styles[law]})" for law in laws),
        fontsize=9,
    )
    panels[-1].set_xlabel("feed flow rate (sccm)")
    figure.savefig(filename, dpi=150)
    print(f"wrote {filename}")


# ------------------------------------------------------------------- the sweep


def flow_sweep(case: Case, feed_rates, **kwargs) -> dict:
    """The abscissa of their Figs. 4 and 5: permeation against feed flow rate."""
    results = {"feed_sccm": list(feed_rates), "permeated": [], "extraction": []}
    for feed in feed_rates:
        point = solve(Case(**{**case.__dict__, "feed_sccm": feed}), **kwargs)
        results["permeated"].append(point["permeated"])
        results["extraction"].append(point["extraction"])
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"  {feed:6.0f} sccm feed -> {point['permeated']:7.3f} sccm permeated"
                f"   ({100 * point['extraction']:5.1f}% of the {point['fed']:.2f} sccm"
                f" of D2 delivered, balance {100 * point['balance']:+.2f}%)"
            )
    return results


def plot(case: Case, sweeps: dict, filename: str) -> None:
    import matplotlib as mpl

    mpl.use("Agg")  # the figure is written to a file, never shown
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 4.2), layout="constrained")
    for label, sweep in sweeps.items():
        axes.plot(sweep["feed_sccm"], sweep["permeated"], "o-", label=label)

    # everything the feed carries, the ceiling any model has to stay under
    feed = np.array(sweeps[next(iter(sweeps))]["feed_sccm"])
    axes.plot(feed, feed * case.d2_fraction, "k--", lw=1, label="all the D2 fed")

    axes.set_xlabel("feed flow rate (sccm)")
    axes.set_ylabel("D$_2$ permeated flow rate (sccm)")
    axes.set_title(
        f"Pd-25Ag tube, {case.temperature - 273.15:.0f} $\\degree$C, "
        f"{case.total_pressure / 1e3:.0f} kPa, "
        f"{100 * case.d2_fraction:.2f}% D$_2$ in He"
    )
    axes.set_xlim(left=0)
    axes.set_ylim(bottom=0)
    axes.legend()
    figure.savefig(filename, dpi=150)
    print(f"wrote {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--temperature", type=float, default=573.15, help="K")
    parser.add_argument("--pressure", type=float, default=90e3, help="Pa, total")
    parser.add_argument("--d2-fraction", type=float, default=0.0395)
    parser.add_argument(
        "--kd",
        action="append",
        choices=sorted(KD_LAWS),
        help="dissociation constant to use; repeat to compare (default: both)",
    )
    parser.add_argument("--n-axial", type=int, default=200)
    parser.add_argument("--n-radial", type=int, default=6)
    parser.add_argument(
        "--fully-developed",
        action="store_true",
        help="floor the Sherwood number at 3.66 instead of using eq. (6) as written",
    )
    parser.add_argument(
        "--quick", action="store_true", help="coarse mesh and three flow rates"
    )
    parser.add_argument(
        "--uhp",
        action="store_true",
        help="check the resistance chain against their pure-D2 eq. (7) and stop",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="reproduce Fig. 4 with the finite element model, refitting k_d, and stop",
    )
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="with --compare: use the plug-flow model instead of FESTIM",
    )
    parser.add_argument(
        "--per-condition",
        action="store_true",
        help="fit k_d to each (T, P) on its own, as their Fig. 6 implies, and stop",
    )
    parser.add_argument("--figure", default="pd_permeator.png")
    args = parser.parse_args()

    if args.uhp:
        if MPI.COMM_WORLD.rank == 0:
            serra_consistency()
            uhp_calibration()
        return
    if args.per_condition:
        # collective: every rank fits its share of the conditions
        fitted = fit_per_condition(reduced=args.reduced)
        if MPI.COMM_WORLD.rank == 0:
            print(f"\nmodel: {'reduced (plug flow)' if args.reduced else 'FESTIM'}")
            report_per_condition_fit(fitted)
        return
    if args.compare:
        # every rank, not just rank 0: compare() makes collective calls
        compare(reduced=args.reduced)
        return

    feed_rates = [100, 400, 1000] if args.quick else [100, 200, 400, 600, 800, 1000]
    mesh_kwargs = {
        "n_axial": 80 if args.quick else args.n_axial,
        "n_radial": 4 if args.quick else args.n_radial,
        "entrance_only": not args.fully_developed,
    }

    sweeps = {}
    for law in args.kd or sorted(KD_LAWS):
        case = Case(
            temperature=args.temperature,
            total_pressure=args.pressure,
            d2_fraction=args.d2_fraction,
            kd_law=law,
        )
        if MPI.COMM_WORLD.rank == 0:
            report(case, mesh_kwargs)
        sweeps[f"$k_d$ {law}"] = flow_sweep(case, feed_rates, **mesh_kwargs)

    if MPI.COMM_WORLD.rank == 0:
        plot(case, sweeps, args.figure)


def report(case: Case, mesh_kwargs: dict) -> None:
    """Prints the dimensionless groups worth knowing before trusting the answer."""
    velocity = gas_velocity(Case(**{**case.__dict__, "feed_sccm": 1000.0}))
    d_gas = binary_diffusivity(case)
    cell_peclet = velocity * (LENGTH / mesh_kwargs["n_axial"]) / d_gas
    k_t = mass_transfer_coefficient(case, entrance_only=mesh_kwargs["entrance_only"])
    k_d = dissociation_constant(case)

    print(f"\n{case}")
    print(f"  D          = {diffusivity(case.temperature):.3e} m2/s")
    print(f"  K_S        = {sieverts_constant(case.temperature):.3e} mol/m3/Pa^0.5")
    print(f"  k_d        = {k_d:.3e} mol/m2/s/Pa")
    print(f"  k_r        = {recombination_constant(case):.3e} m4/mol/s")
    print(f"  K_T        = {k_t:.3e} m/s  (Sh = {k_t * 2 * R_IN / d_gas:.2f})")
    print(
        "  gas-phase share of the feed-side resistance: "
        f"{k_d * F.R * case.temperature / k_t:.1%}"
    )
    print(f"  cell Peclet at 1000 sccm: {cell_peclet:.2f}")


if __name__ == "__main__":
    main()
