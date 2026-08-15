# Pd-25Ag permeator

A tubular palladium-silver membrane extracting deuterium from a helium stream, after
[Fuerst, Taylor & Shimada (2024)](https://doi.org/10.1109/TPS.2024.3356857) — the PreTEX
experiment at INL.

```bash
python pd_permeator.py --quick        # ~1 min, three flow rates
python pd_permeator.py                # the full flow sweep for both k_d laws
```

## What it demonstrates

The measured permeated flow rate rises with the feed flow rate and then plateaus. That
rise is axial depletion of D₂ in the feed: at 100 sccm the stream only carries 3.95 sccm
of D₂ and the tube extracts most of it. Capturing it means transporting the feed gas as
its own field along the membrane, so the model is

- the **membrane** as an ordinary 2D subdomain, `x` across the wall and `y` along it;
- the **feed gas** as a codim-1 `VolumeSubdomain(dim=1)` running along the inner face,
  with an `AdvectionTerm` carrying it down the tube;
- **inlet and outlet** as codim-2 `SurfaceSubdomain(dim=0)`, one with a
  `FixedConcentrationBC` and one with an `OutflowBC`;
- the **gas/metal exchange** as a `ParticleFluxBC` and a `ParticleSource` sharing one
  `species_dependent_value` expression, so each half sees both concentrations;
- **recombination into the vacuum** as a flux on the outer face.

Which is to say it needs essentially every part of the codim-1 and drift machinery at
once, and it is a case where a model without them gets the answer qualitatively wrong.

## Things worth knowing

**Units are moles of D₂**, as in the paper, so Table I can be copied verbatim. That is
why desorption is written as a plain `ParticleFluxBC` rather than with
`SurfaceReactionBC`, whose factor of two assumes the atomic convention.

**The two halves of the exchange are per different things.** The wall gains a flux per
unit membrane area; the gas loses it per unit gas volume. For a tube the conversion is
the perimeter-to-area ratio `2/r_i`. Getting it wrong shows up immediately as a mass
balance that does not close, which the script prints for every point.

**So are the exported fluxes.** A `SurfaceFlux` on a wall facet multiplies back up by the
circumference `2πr_i`; one at the end of the *manifold* multiplies by the flow
cross-section `πr_i²`, because the manifold's own equation is written per unit gas
volume.

**Newton needs a starting point.** The desorption terms go as `-k_r c²`, whose derivative
vanishes at `c = 0`, so from an all-zero guess the first step overshoots by ten orders of
magnitude and takes ~50 iterations to recover. A steady problem takes no initial
condition, so the guess is written into the solution vector directly — see
`seed_newton()`.

**Digitising is programmatic, not by hand.** `digitise_fig4.py` and `digitise_fig5.py`
rasterise the PDF, threshold on hue, fill the open markers and keep only what holds a
larger inscribed disk than a curve can — which separates the data points from the model
lines they sit on. Fig. 4 needs two extras: each of its four panels is calibrated
separately (their frames span 937–948 px for the same 0–1200 sccm, so one shared
calibration puts the top panel out by ~18 sccm), and the grey series has to be masked
away from black text, whose antialiasing otherwise passes as mid-grey. The acceptance
test is built in: the campaign stepped the feed by 100 sccm, and every recovered point
lands within 3.4 sccm of a multiple of it.

**The tube is unrolled into a plane**, since codim-1 subdomains are cartesian-only. The
wall thickness used is the cylindrical resistance referred to the inner surface,
`r_i ln(r_o/r_i)` = 74.2 µm rather than the geometric 76.2 µm, so the planar model
reproduces the cylindrical flux per unit inner area exactly.

**The inlet is a Dirichlet condition, not a Danckwerts one.** Because the concentration
falls downstream, diffusion carries D₂ *forwards* across the inlet and the model receives
a few percent more than the stream physically delivers (3% at 100 sccm, under 1% at
1000 sccm). Fixing the total flux instead would be right, but the boundary of a manifold
currently only accepts a `FixedConcentrationBC`.

## Agreement with the paper

```bash
python digitise_fig4.py                            # rebuild the CSV from the PDF
mpirun -np 16 -x OMP_NUM_THREADS=1 \
    python pd_permeator.py --compare               # the finite element model
mpirun -np 16 -x OMP_NUM_THREADS=1 \
    python pd_permeator.py --compare --reduced     # the same, via plug flow
```

`mpirun` here is not the usual FESTIM kind. `--compare` is hundreds of *small unrelated*
solves — the optimiser walks 102 operating points per residual and needs tens of those —
so `map_solve()` builds each rank's mesh on `MPI.COMM_SELF` and deals the cases
round-robin across `COMM_WORLD`, then allgathers. Instead of one problem partitioned over
16 ranks there are 16 independent serial problems. FESTIM takes every communicator it
needs from `mesh.comm`, so nothing else has to know. A single operating point still runs
the ordinary way — pass `comm=MPI.COMM_WORLD` to `build()` — it just isn't worth it on a
150 × 4 mesh.

**Pin the threads.** Each rank is a whole serial problem, and its linear algebra will
open a thread per core: 34 per rank in one measurement, so 16 ranks put 544 threads on 16
cores, load average hit 120 and the run took 7× longer than it should. The ranks *are*
the parallelism. This does not arise in an ordinary FESTIM run, where each rank holds
only a slice of one mesh.

Fig. 4 — the whole mixed-gas campaign, four temperatures × four pressures — has been
digitised by colour into [fig4_digitised.csv](fig4_digitised.csv): 102 points over 16
conditions. ([fig5_digitised.csv](fig5_digitised.csv) holds the two series of Fig. 5,
which carry the DM/CM/CM-O model lines; where the two figures overlap, prefer Fig. 4.
Their ordinates agree to ~0.1 sccm but Fig. 4 puts every point on the 100 sccm steps the
campaign used, to within 3.4 sccm, while Fig. 5 does not.)

Refitting `k_d` to all of them with the finite element model, as the paper itself did
with theirs (~15 min on 16 ranks):

```
model: FESTIM
k_d refitted to 102 points: 4.432e-05 exp(-16810/RT)  [their eq. (9): 7.99e-04 exp(-27600/RT)]

rms disagreement with the 102 digitised points
  k_d serra       81.8%
  k_d optimised   36.0%
  k_d fitted      10.9%

by condition, k_d fitted, worst first
  300 C  150 kPa  ( 5 pts)   29.5%      400 C  250 kPa  ( 9 pts)    5.6%
  300 C   90 kPa  ( 7 pts)   21.4%      450 C  250 kPa  (10 pts)    5.2%
  350 C   90 kPa  ( 5 pts)   15.5%      350 C  150 kPa  ( 5 pts)    4.6%
  300 C  250 kPa  ( 8 pts)   10.4%      400 C  190 kPa  ( 7 pts)    3.4%
  300 C  190 kPa  ( 6 pts)    9.1%      450 C  150 kPa  ( 5 pts)    3.0%
  400 C  150 kPa  ( 7 pts)    8.6%      450 C  190 kPa  ( 5 pts)    2.7%
  400 C   90 kPa  ( 6 pts)    8.2%      450 C   90 kPa  ( 6 pts)    2.3%
  350 C  190 kPa  ( 5 pts)    5.8%      350 C  250 kPa  ( 6 pts)    1.1%
```

The model form reproduces the experiment across the whole grid; the constant does not
carry over, coming out a factor 0.53 (300 °C) to 0.33 (450 °C) below their published
`k_d`.

Sixteen conditions pin two parameters far harder than Fig. 5's two did — those alone fit
to 5.3% — and the residual is **structured, not scattered**: it grows steadily as the
temperature falls, and within each temperature is worst at the lowest pressures. That is
exactly the corner where the surface resistance dominates, which is the same finger
pointed at eqs. (2)–(5) below.

Some of the 300 °C residual isn't the model's: in that panel the measured 150 kPa series
sits **above** the 190 kPa one, which no monotonic model can reproduce and which is not
the ordering the other three temperatures show.

Three other checks:

- **Their pure-D₂ campaign** (`--uhp`) is the one part of the data in closed form, the
  permeability fit of eq. (7). Their optimised `k_d` reproduces it to 9–11% at all four
  temperatures. Fitting `k_d` to *that* campaign instead gives `2.5e-4 exp(-23626/RT)`,
  ~1.8× what the mixed-gas curves want — the two datasets don't quite agree on `k_d`
  through this model.
- **An independent solve.** `plug_flow()` integrates the same physics as a 1D gas balance
  with a local flux at each station — the structure of the paper's own 50-node code,
  sharing nothing with the finite element model but the material properties. Across all
  102 conditions the two differ by **−0.29% on average, 2.67% at worst**, the largest
  gaps where extraction is strongest and the Dirichlet inlet strains hardest. Refitting
  through it (`--compare --reduced`) moves `k_d` by 2% in the prefactor and 0.4% in the
  activation energy, and the rms not at all.
- **Internally.** Mass balance closes to better than 0.3% (0.7% at the most strongly
  extracting point); mesh converged to four digits at 150 axial × 4 radial cells.

- **Serra et al. (1998)**, where every metal property originates, also checks out
  (`--uhp`, first table): `sqrt(k_d/k_r)` returns their `K_S` to 0.8%, `D·K_S` returns the
  permeability they measured directly to 0.3%, Fuerst's Table I is transcribed from them
  faithfully, and in the surface-limited limit the chain converges on their eq. [4],
  `J = k_d·p/2`, to four digits. Serra measure J by pressure rise and state it in "moles
  of gas", with `c = K_S p^½` in the same basis — so the convention is molecular
  throughout and **there is no factor of two hiding between D₂ and D atoms**.

### The part that is not explained

Fig. 5 also draws the DM and CM lines, so those compare model to model. The **DM agrees**
— 12.9 sccm here against ~13.0 off the figure, at 300 °C, 90 kPa, 1000 sccm — which pins
down geometry, solubility, diffusivity, axial depletion and the sccm conversion. The
**CM does not**: 11.0 sccm here against ~6.3. The two models differ only by the surface
terms, and those are now verified against Serra directly, so the difference is in how
eqs. (2)–(5) are implemented on their side, not in the constants and not in the
convention.

One thing the papers leave open: Serra obtained `k_1` by fitting J/p through *their*
eq. [5], a series interpolation between the diffusion- and surface-limited regimes. A
constant carried out of that fit into a directly solved resistance chain need not mean
quite the same thing. That is a question for the authors rather than something to patch
here.
