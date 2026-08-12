# Deuterium transport in a proton-conducting ceramic — FESTIM vs TMAP8

Reimplementation of

> L. Yang, P.-C. A. Simon, W. Tang, M. Li, Z. Zhao, D. Ding, T. Fuerst,
> *Elucidating hydrogen isotope transport mechanisms in proton-conducting ceramics
> with trapping effects using TMAP8*, Int. J. Hydrogen Energy **210** (2026) 153551

published as TMAP8 validation case
[val-2g](https://github.com/idaholab/TMAP8/tree/devel/test/tests/val-2g).

| file | what |
|---|---|
| `pcc_bzy_tds.py` | the FESTIM model, runs both environments (~4 min) |
| `pcc_bzy_dry.png`, `pcc_bzy_wet.png` | TDS spectra + inventories |
| `paper_figures/` | figures extracted from the paper PDF |

Note: the six Fig. 4 panels are **not** embedded in the PDF in reading order. They
were identified by their RMSPE fingerprints and renamed to their true (a)–(f) labels.

## The model

Deuterium TDS from a 0.5 mm BaZr₀.₉Y₀.₁O₂.₉₅ (BZY) membrane: 3600 s dissolution at
873 K under D₂ (dry) or D₂O (wet), quench to 300 K, pump down, then a 0.5 K/s ramp
to 1400 K in vacuum.

- **Three mobile species**, each with its own Arrhenius diffusivity: the hydroxyl
  OD⋅ that carries the deuterium, oxygen vacancies V_O⋅⋅, electrons e′.
- **One trap population**, Eq. (2):
  `dC_t/dt = α_t (χN − C_t) C_OD / N − α_r C_t`
- **Two reversible kinetic surface reactions** on both faces, Eqs. (6)–(9):
  ```
  D2O + V_O.. + O_O^x <-> 2 OD.         r_w = K1w·P_D2O·C_V·C_Ox − K-1w·C_OD²
  D2  + 2 O_O^x       <-> 2 OD. + 2 e'  r_d = K1d·P_D2·C_Ox²    − K-1d·C_OD²·C_e²
  ```
  Both run in **both** environments — only the loading pressures differ. That is why
  the paper reports a D₂ *and* a D₂O flux per environment.
- Lattice oxygen is not an independent unknown: the oxygen sublattice is shared,
  `C_Ox = 3N − C_V − C_OD`.

## Mapping onto FESTIM

Everything needed already exists.

| paper / TMAP8 | FESTIM |
|---|---|
| multi-species diffusion with per-species D(T) | `Species` × 3 + `Material(D_0={...}, E_D={...})` |
| trapping, Eq. (2) | `ArrheniusReaction(reactant=[OD, ImplicitSpecies(n=χN, others=[trapped])], product=trapped, k_0=τ_t0/N, ...)` |
| lattice oxygen `3N − C_V − C_OD` | expression in the flux (or `ImplicitSpecies`) |
| kinetic surface reactions, Eqs. (8)–(9) | `ParticleFluxBC(..., species_dependent_value={...})` |
| T(t), P(t) histories | `temperature=lambda t: ...`, `ufl.conditional` on the time constant |
| adaptive dt with milestones | `Stepsize(growth_factor=..., milestones=[...])` |
| D₂ / D₂O release for the TDS spectrum | `CustomQuantity` per surface |

The one awkward part is the surface kinetics — see [the gap](#the-festim-gap) below.

## Result: all four channels reproduce

| channel | FESTIM | measured (Fig. 4e,f) |
|---|---|---|
| dry, D₂ | 1.33×10⁻¹¹ mol/s @ 1028 K | 1.48×10⁻¹¹ @ ~1010 K |
| dry, D₂O | 6.01×10⁻¹² @ 1078 K | 6.3×10⁻¹² @ ~1100 K |
| wet, D₂O | 2.01×10⁻⁹ @ 990 K | 1.93×10⁻⁹ @ ~1010 K |
| wet, D₂ | 6.83×10⁻¹¹ @ 944 K | ~7×10⁻¹¹ @ ~1000 K |

Within ~10% on magnitude and ~20 K on peak temperature throughout, with curve shapes
that overlay the published TMAP8 curves.

## Finding: deuterium is not conserved in TMAP8's wet system

In `val-2g_trapping.i` the dry and wet flux blocks are otherwise character-for-character
mirrors, differing in exactly one line:

```
flux_on_OT_dry = 2 * flux_base_on_T2_dry + 2 * flux_base_on_T2O_dry
flux_on_OT_wet = 2 * flux_base_on_T2O_wet
```

`flux_base_on_T2_wet` is the D₂ reaction rate. With `pressure_T2_wet = 0` it is purely
negative — the rate at which D₂ recombines and leaves the surface. Each such molecule
should remove 2 OD⋅ and 2 e′ from the solid. Per D₂ molecule formed in the wet system:

| | dry | wet |
|---|---|---|
| 2 e′ consumed | ✅ `flux_on_e = 2*flux_base_on_T2` | ✅ same |
| 2 OD⋅ consumed | ✅ in `flux_on_OT` | ❌ **absent** |
| D₂ reported as released | ✅ `flux_on_T2 = -flux_base_on_T2` | ✅ same |

So deuterium appears in the reported gas flux without leaving the solid, and charge
balance breaks too (e′ debited, the charged OD⋅ not). That reported flux is not inert:
it is compared against the measured D₂ data and assigned RMSPE = 40.49% in Fig. 4(f).

**It is load-bearing.** Hydration loads far more deuterium than dry dissociation
(~0.103 vs ~0.0013 mol/m² of mobile OD here, ~80×), and the D₂ recombination rate goes
as `C_OD²·C_e²`, so at 80× the concentration that channel runs ~6000× faster and becomes
the cheapest way out of the solid. Running both ways, everything else identical:

- **omitted** (their convention): D₂O peak 2.01×10⁻⁹ mol/s @ 990 K — matches Fig. 4(f)
- **restored** (deuterium conserved): D₂ takes the whole release, 9.13×10⁻¹⁰ @ 1172 K,
  and the D₂O peak disappears

`TMAP8_WET_OT_FLUX_OMITS_D2` in `pcc_bzy_tds.py` switches between them. The dry case is
complete and unaffected.

**Is it a bug?** The conservation violation is not a judgement call — the atoms do not
balance, and the term that would balance them sits ten lines above in the sibling block.
Intent is unknowable from here, but it reads as an oversight: the blocks are copy-paste
twins differing by one addend, and "keep the electron coupling and keep reporting the
flux, but drop the hydroxyl sink" is not a coherent simplification of anything.

Caveats: the counterfactual comes from a close reimplementation (~10% agreement), not
from their code; and `val-2g_main_PSS_trapping.i`, used for the optimisation, was not
inspected. The decisive test is one line — add `+ 2 * flux_base_on_T2_wet` to
`flux_on_OT_wet` and rerun. Worth filing as a TMAP8 issue: it bears on the paper's claim
of a single unified model across dry and wet environments.

## Other things the source resolved

Several quantities cannot be reconstructed from the paper alone; the `.params` files
settle them.

- **Exponent conventions.** The tabulated "exponents" are not `10^x` — the prefactors
  are `χ = 3×10^x`, `τ_t0 = 4.8×10^x`, `τ_r0 = 2.6×10^x`, `K1 = 2×10^x`, and
  `C_e′₀ = 10^x · N`. These recover the paper's quoted values (χ = 8.3×10⁻³,
  τ_t0 = 3.96×10⁹ s⁻¹, τ_r0 = 2.06×10¹⁸ s⁻¹).
- **C_e′₀ is a lattice fraction**, not the mass density its "g/cm³" unit suggests:
  535 mol/m³, not ~10⁴. Since the D₂ reverse rate goes as `C_e²`, getting this wrong
  by 34× makes that channel ~1200× too fast and sends the entire wet release out as D₂.
- **Units.** TMAP8 runs val-2g in at/µm³ and µm. Forward rates are unit-invariant under
  `K1 × N_A` (both terms quadratic in concentration), but the reverse D₂ term is quartic
  (`C_OD²·C_e²`), so `K_eq` of Eq. (11) — which carries concentration² — must be
  converted to the frame it was calibrated in. Hence `DRY_KEQ_UNIT` in the script.
- **Sign of Eqs. (4)–(5).** The paper writes `α = τ₀ exp(+ε/k_BT)`. `TrappingNodalKernel`
  computes `α_t·exp(−ε_t/T)·(χN − C_t)·C_m/N`, i.e. the **negative** exponent. The
  published sign is a typo.
- **Miscellany.** `C_V0 = 0.05 N` (`hydration_limit_S/2`); `A = 7.7 × 2.2 mm`; cooldown
  is exponential with τ = 600 s; pumping at 6200 s; the hydroxyl diffusivity prefactor
  is scaled by `√(3/2)` (the case is written for tritium, the data is deuterium); fluxes
  are reported on the left face only.

## The FESTIM gap

`SurfaceReactionBC` is hardcoded to `K = k_r·∏C_i − k_d·P` — one gas pressure, no species
on the dissociation side, and the same flux for every reactant. These reactions need
`C_V`/`C_Ox`/`C_e` in the rate and a different stoichiometric coefficient per species, so
they are written by hand as ~40 lines of `ParticleFluxBC` with `species_dependent_value`.

A `GenericSurfaceReaction` mirroring the volume `GenericReaction` would collapse that to:

```python
O_x = F.ImplicitSpecies(n=3 * N_SITE, others=[V_O, OD], name="O_O^x")

F.GenericSurfaceReaction(          # D2O + V_O.. + O_O^x <-> 2 OD.
    subdomain=surface,
    reactant=[F.Gas(pressure=p_d2o), V_O, O_x],
    product=[OD, OD],
    forward_rate=k1_wet,
    backward_rate=lambda T: k1_wet / K_eq_wet(T),
)
F.GenericSurfaceReaction(          # D2 + 2 O_O^x <-> 2 OD. + 2 e'
    subdomain=surface,
    reactant=[F.Gas(pressure=p_d2), O_x, O_x],
    product=[OD, OD, e, e],
    forward_rate=k1_dry,
    backward_rate=lambda T: k1_dry / K_eq_dry(T),
)
```

The semantics all exist in `GenericReaction` already: `R = k₁∏reactants − k₂∏products`;
stoichiometry by repetition (`[OD, OD]` both squares OD in the backward term and gives it
`+2R` of flux); reactants get `−R` and products `+R`, matching FESTIM's influx-positive
convention; implicit species contribute to the rate but receive no flux. The only new
piece is a gas participant carrying a pressure instead of a concentration — and
`SurfaceReactionBC` already accepts a float, a callable or an `F.GasSpecies` there, so
passing a `GasSpecies` would feed an enclosure mass balance for free.

Incidentally, in this form the conservation asymmetry above is close to unwriteable by
accident: every species draws its flux from the same `R`.

## Running it

```bash
python pcc_bzy_tds.py        # -> pcc_bzy_dry.png, pcc_bzy_wet.png
```

Switches at the top of the file: `PARAMS["supp"]` (literature values) vs
`PARAMS["calibrated"]` (the 18-parameter PSS calibration behind Fig. 4e,f);
`TMAP8_WET_OT_FLUX_OMITS_D2`; `P_VAC` (residual pressure after pumping).
