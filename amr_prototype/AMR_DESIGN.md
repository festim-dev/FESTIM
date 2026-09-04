# Adaptive mesh refinement in FESTIM — investigation notes

**Status:** investigation. Nothing in `src/festim/` is modified. Everything here is a
prototype driving FESTIM from the outside, plus the design it argues for.

**Environment:** DOLFINx 0.11.0, FESTIM `rem/drift`. Target problem class is
`HydrogenTransportProblemDiscontinuous`.

---

## 1. What DOLFINx can and cannot do

### It cannot coarsen

`dolfinx.mesh` exposes `refine`, `uniform_refine` and `transfer_meshtag`. Plaza
refinement is one-way and there is no inverse. **Every form of unrefinement has to come
from rebuilding a mesh**, which forces the whole design.

Two ways to rebuild, and they are the two backends in `adapters.py`:

| | `Mesh1DAdapter` | `BaseMeshRefiner` |
|---|---|---|
| how | rebuild the vertex array outright | keep the base mesh forever, re-refine it against a per-base-cell *level* field |
| coarsening | exact, arbitrary | back to base resolution, in whole levels |
| dimensions | 1D only | any |
| parent map | none | composed across refine passes |

`BaseMeshRefiner` was verified to coarsen properly: moving a refined band from
x ∈ [0, 0.3] to x ∈ [0.6, 0.8] on an 8×8 unit square returned the left region to
*exactly* the base cell size (h = 0.1767766952966369, bit-identical to the base mesh)
while the new band reached h = 0.022.

### `parent_cell` is not an index into the refined mesh

`refine` returns `(mesh, parent_cell, parent_facet)`. `parent_cell` looks like a
cell-indexed array — `[0 0 0 0 1 1 1 1 2 2 2 2 ...]` — and it is not one. **DOLFINx
reorders cells after refining, and the returned array refers to the pre-reordering
order.** Checking every child midpoint against its claimed parent triangle on a 4×4 unit
square, `parent_cell` agreed with the true parent for **0%** of cells, and children were
placed up to 1.2 length units from their claimed parent in a domain 1 unit across.

`transfer_meshtag` accounts for the reordering internally and is correct. So the only
reliable way to carry per-cell data across a refinement is to put it in a meshtag and
transfer it — which is what `BaseMeshRefiner` does to maintain its ancestry map:

```python
ancestry_tag = dmesh.meshtags(base, vdim, arange(n_base), arange(n_base))
...
ancestry_tag = dmesh.transfer_meshtag(ancestry_tag, refined, parent_cell)
```

This one cost the most debugging time of anything in the investigation, because the
failure is silent: the mesh is valid, the solve converges, and the refined region simply
drifts to the wrong place.

### 1D refinement is crippled

`RefinementOption.parent_cell_and_facet` raises
`RuntimeError: Parent facet computation not yet supported!` on interval meshes.
`parent_cell` alone works. So **facet meshtags cannot be transferred in 1D at all**, and
`BaseMeshRefiner` is unusable in 1D whenever the tags are user-supplied.

### `transfer_meshtag` needs connectivity on the refined mesh

Otherwise: `RuntimeError: Refined mesh is missing cell-facet connectivity`. Call
`create_entities(1)` plus both cell↔facet `create_connectivity` on the refined mesh
first.

---

## 2. What FESTIM makes easy, and what it makes hard

### Easy: meshtags are geometric

`Mesh.define_meshtags` ([`mesh.py:130-262`](../src/festim/mesh/mesh.py)) re-derives every
tag from locators (`locate_boundary_facet_indices`, `locate_subdomain_entities`) with no
index caching. It is safe to re-run on any new mesh. Combined with `Mesh1D` building from
an arbitrary sorted vertex array, a 1D remesh is just `F.Mesh1D(new_vertices)`.

### Easy: the state transfer already exists

- `helpers.nmm_interpolate` ([`helpers.py:433-460`](../src/festim/helpers.py)) wraps
  `create_interpolation_data` + `interpolate_nonmatching`.
- The discontinuous `create_initial_conditions` already accepts a `fem.Function` and
  routes it through `nmm_interpolate`
  ([`hydrogen_transport_problem.py:1710-1711`](../src/festim/hydrogen_transport_problem.py)).

That second point is what makes a zero-FESTIM-change prototype possible: a remesh is a
*restart* with the previous solution as the initial condition.

### Hard: `initialise()` cannot be re-run in place

Re-running it on a new mesh hits, in order:

- `create_species_from_traps` **appends** to `self.species` and `self.reactions`, so a
  second call duplicates every trap;
- `define_meshtags_and_measures` short-circuits unless `facet_meshtags` and
  `volume_meshtags` are both reset to `None`
  ([`problem.py:157-161`](../src/festim/problem.py));
- `initialise_exports` **truncates** the output files;
- the VTX / XDMF / VTKHDF writers are bound to specific `Function` objects and write the
  mesh once, so a new mesh produces an inconsistent file;
- `Profile1DExport._dofs`, `export.D` / `D_expr`, `export.volume_meshtags`,
  `export.facet_meshtags` and `CustomQuantity.ufl_expr` are cached on first use and never
  invalidated.

The prototype sidesteps all of it by building a **fresh problem object** on the new mesh
(`restart.py`). Everything in the discontinuous `initialise()` — submeshes,
`cell_map`/`v_map`/`n_map`, interface data, `surface_to_volume`, the blocked solver —
derives from `self.mesh.mesh` plus locators, so a fresh build is correct by construction,
and the ffcx cache makes the repeated form compilation nearly free after generation one.

One catch found the hard way: `create_initial_conditions` only fills `u_n`. Leaving `u`
at zero hands Newton the worst possible first guess and makes every post-processed
quantity read zero until the first solve lands. `restart.seed_current_from_previous`
copies `u_n → u` and refreshes the collapsed post-processing functions.

### Hazard: an absolute solver tolerance does not survive refinement

The 2D demo silently produced a *wrong steady answer* until this was found, and it is the
most dangerous thing in these notes because nothing reports an error.

A SNES absolute tolerance is compared against a residual that carries a factor of the
cell measure. The 1D demo runs happily at `atol=1e12` because a 1D cell has length
10⁻⁶. A 2D cell of the same nominal size has area ~5·10⁻¹³ — six orders of magnitude
smaller — so at `atol=1e12` the *entire* trapped-species residual sat below the
tolerance. SNES converged on the first iteration, `iterate()`'s convergence assert
passed, and the trapped concentration froze at exactly `0.1 n_trap` for 10⁵ s while the
true local equilibrium was `n_trap`. No front ever formed.

```
atol=1e+12: t=7098  ct_max/N=0.1000  saturated area=0.00000   <- wrong, no error raised
atol=1e+06: t=3228  ct_max/N=1.0692  saturated area=0.05812
atol=1e+02: t=1057  ct_max/N=1.0857  saturated area=0.03000
```

**AMR makes this strictly worse.** Every refinement shrinks the cell measure again — three
levels in 2D is a factor of 64 — so a tolerance that was adequate on the base mesh
becomes progressively more permissive exactly as the mesh gets better. A first-class AMR
implementation should either scale `atol` with the mesh or refuse to accept a
user-supplied absolute tolerance without warning once adaptation is enabled.

### Hazard: subdomain borders and floating point

`np.linspace(0, 1e-4, 101)[50]` is `4.9999999999999996e-05`, not `5e-05`. A
`VolumeSubdomain1D` with `borders=[L/2, L]` uses `x >= L/2` as its locator, so the cell
straddling that vertex is tagged by neither subdomain and `define_meshtags` fails its
"all cells must be tagged" assert. This exists today, but AMR makes it far more likely
because vertex coordinates become *generated* rather than written by a user. Hence
`Mesh1DAdapter` treats every subdomain border as a protected vertex and reproduces it
exactly rather than recomputing it.

---

## 3. Choosing what to refine

### The relative/absolute distinction matters more than the estimator

The DOLFINx tutorial uses a residual-flavoured indicator with Dörfler marking: refine the
cells whose indicator exceeds `θ · max(η)`. **That criterion never terminates**, because
there is always a top 15%. In the transient demo it drove the mesh from 215 to 1719 cells
over the run while the front itself stopped getting sharper — the mesh was chasing its own
maximum.

For a transient problem the criterion has to be *absolute*, so the mesh reaches a steady
size once the front is resolved. `variation_indicator` measures the relative change of a
species across a cell:

> η_K = (max_K c − min_K c) / max_Ω c

which reads as "the concentration changes by 5% across this cell". `threshold_mark`
refines above 5% and coarsens below 0.2%, and the mesh settles.

So: **Dörfler for steady AMR driven to a tolerance, absolute thresholds for transient
front-tracking.** Both are in `indicators.py`.

### Read the dofmap, don't assemble a form

The first `variation_indicator` assembled a DG0 form. It cost **61 ms per adapt**, several
times the 20 ms timestep it was deciding about, and was the single largest cost in the
run. For a P1 field the spread of vertex values over a cell *is* `h_K|∇u|`, so it reads
straight off `V.dofmap.list`: **0.27 ms**, a 225× reduction.

### Which species you measure decides the mesh

In the 2D demo, marking on *both* species kept the entire filled region refined and the
mesh never came back to base resolution behind the front. The trapped field is flat at
saturation there, but the mobile field is not: hydrogen still diffuses across the filled
region to feed the front, and its relative change across a base cell (~0.08) lands
between `coarsen_below` and `refine_above`, so those cells are never told to do anything
and keep whatever level they were given when the front passed over them.

Marking on `trapped` alone gives a clean arc with base resolution on both sides of it,
and 24970 cells instead of 28203 for the same run.

Neither answer is wrong -- the mobile gradient *is* real -- but it means the species list
is a modelling decision the user has to make, not something a library can default well.
`problem_indicator(species_names=[...])` exposes it.

### Normalise per species

Mobile and trapped concentrations differ by four orders of magnitude here. Without
per-species normalisation a single trap decides the entire mesh.

### A moving front needs a buffer

The front moves between two adapts, and without a buffer it walks straight out of the
refined band: at late times it travelled **1.4 µm per adapt**, many times the refined cell
size, leaving the resolved band behind it and producing a one-cell jump from 1.13 to 0.17
of saturation. `Mesh1DAdapter(buffer=...)` also refines within a given *distance* of a
marked cell. Distance, not a cell count — the cells near a front are precisely the small
ones, so a fixed cell count shrinks exactly where the front is about to move.

### Split more than once per adapt

A cell four times over threshold needs two bisections. Doing one per adapt made the mesh
take ~70 timesteps to catch up with the initial boundary layer. `threshold_mark` returns
`ceil(log2(η/refine_above))` splits, clipped to `max_splits`.

---

## 4. Results

### Steady state — a clear win

Slab with a narrow Gaussian implantation source (2·10⁻⁷ m wide in a 10⁻⁴ m slab); QoI is
the outgassing flux at the left surface. `demo_steady.py`:

```
iteration  0:     40 cells   flux = 4.94915563e+19   change = inf
iteration  1:     42 cells   flux = 7.18518547e+19   change = 3.11e-01
iteration  2:     44 cells   flux = 9.90854672e+19   change = 2.75e-01
iteration  3:     47 cells   flux = 9.89945523e+19   change = 9.18e-04
iteration  4:     50 cells   flux = 9.89959002e+19   change = 1.36e-05

uniform    640 cells: flux = 9.89996669e+19
uniform  10240 cells: flux = 9.89999674e+19
```

**50 cells reach the 10240-cell uniform answer to 0.004%** — a 200× reduction in cells.

### Transient trapping front — a quality win, not a speed win

Tungsten slab at 300 K, fixed mobile concentration on the left, deep traps that fill
essentially irreversibly, so the trapped profile is a step advancing as √t.
`demo_transient_trapping_1d.py --mode all`, final time 10⁵ s:

```
     run        cells     wall   inventory     front  overshoot  profile L2
  coarse      100-100     3.5s      0.08%    0.07%    15.12%      1.65%
     amr      100-943     5.0s      0.22%    0.22%    -0.00%      1.13%
    fine    2000-2000     4.2s      0.00%    0.00%    -0.00%      0.00%
```

AMR eliminates the overshoot above trap saturation (15.1% → 0.0%) and halves the profile
error, using at most 943 cells against 2000 — **but it is slower than the fine mesh.**

Over three repeats the AMR run took 6.89, 5.23, 4.88 s against 4.23, 4.31, 4.35 s for the
fine mesh: consistently ~15-20% slower once the ffcx cache is warm, and the first run pays
an extra ~1.7 s compiling the extra form variants the rebuilt problems need.

That is structural, not a tuning failure. Compare coarse and fine: **20× the cells costs
20% more wall time.** The 1D solve is dominated by fixed per-step overhead (form assembly,
SNES setup), not by the linear algebra that AMR shrinks, so there is almost nothing for
AMR to recover. On top of that the prototype's rebuild is not free:

```
     solve   2.87s  70.6%      20 ms per timestep
 indicator   0.01s   0.2%
     adapt   0.05s   1.2%
   rebuild   1.14s  28.0%      39 ms per adapt
```

**Conclusion: in 1D, AMR buys correctness and robustness, not speed.** Robustness is not a
small claim — with the *physical* trapping rate (front width ≈ 1 nm) the uniform-mesh run
oscillated to `c_t = 1.39 n_trap` and `−0.48 n_trap` and the Newton solve diverged
outright. The demo had to soften `K_0` from 10⁻¹⁷ to 10⁻²² m³/s to have any resolvable
front at all.

A speed win needs the nD backend, where the solve actually scales with cell count.

### 2D curved front — the win the 1D case could not show

Hydrogen enters through a patch on the left edge of a 40 µm square of tungsten, so the
saturated region grows as a *curved* front and the refined region is an arc that has to
migrate. `demo_2d_front.py --compare`, final time 10⁵ s, base mesh 30x30 with
`max_level=3` (so the AMR run reaches h = 1.7·10⁻⁷ at the front, the same cell size as
the 240x240 uniform mesh):

```
                       run            cells     wall  area err  overshoot
                       amr       1800-24970    50.4s     0.64%      0.21%
   uniform base (h=1.3e-6)             1800    14.6s     0.75%     37.67%
   uniform fine (h=1.7e-7)           115200   162.9s     0.00%      0.57%
```

**At the same front resolution, AMR uses 4.6x fewer cells and runs 3.2x faster**, and its
overshoot above trap saturation is if anything *better* than the uniform fine mesh's
(0.21% vs 0.57%), because it puts its resolution where the front actually is at each
instant. The uniform base mesh is 3.4x faster still and completely wrong — 37.7%
overshoot.

This is the comparison 1D could not produce. There, 20x the cells cost 20% more wall
time, so there was nothing to reclaim. Here the solve cost tracks the cell count, and
the cell count is what AMR controls.

Note that *filled area* barely separates the three runs (0.64% / 0.75% / 0.00%) -- it is
an integral, and integrals stay accurate on meshes that badly misrepresent the front.
The overshoot is the metric that discriminates, in 2D as in 1D.

The interior does come back to base resolution behind the front (see
`out/front_2d_amr.png`): the mesh is a migrating band, not a growing one. The cell count
still climbs over the run, but only because the arc gets longer as its radius grows.

### Conservation

Refining is exact: every new vertex sits where the old P1 field is linear
(`test_refinement_transfer_is_exact` asserts this to 1e-12). **All the drift comes from
coarsening**, and it is systematic — inventory *gains* about 1.8·10⁻⁴ per remesh. The
controls trade off cleanly:

| refine_above | coarsen_below | buffer | max cells | total drift |
|---|---|---|---|---|
| 0.05 | 0.01 | 1.5 µm | 467 | 5.1·10⁻³ |
| 0.05 | 0.01 | 3 µm | 755 | 1.6·10⁻³ |
| 0.05 | **0.002** | 3 µm | 943 | **1.8·10⁻⁴** |
| 0.02 | 0.004 | 3 µm | 966 | −4.4·10⁻⁵ |

Tightening `coarsen_below` from 1% to 0.2% cuts the drift 10× for 25% more cells. For an
inventory-critical run that is the knob to turn.

A proper fix would be an L2 projection instead of pointwise interpolation. It was not
prototyped because it needs the old function at the new mesh's *quadrature* points, and
DOLFINx's non-matching machinery only evaluates at *interpolation* points. A global
rescale to conserve the integral is not a substitute here: it would push the saturated
plateau above `n_trap` and re-introduce exactly the instability AMR is being used to
avoid.

---

## 5. Proposed FESTIM API

```python
model.settings = F.Settings(
    ...,
    amr=F.AMR(
        indicator="variation",  # or a callable(problem) -> per-cell array
        refine_above=0.05,
        coarsen_below=0.002,
        hmin=5e-9,
        hmax=2e-6,
        buffer=3e-6,
        max_cells=10_000,
        adapt_every=5,
    ),
)
```

Hook point is right after the `u_n ← u` copy in `iterate()`
([`problem.py:339`](../src/festim/problem.py), discontinuous equivalent at
[`hydrogen_transport_problem.py:3437-3440`](../src/festim/hydrogen_transport_problem.py)),
calling a new `ProblemBase.remesh(new_mesh)`.

### Work required to get there

1. **Make `initialise()` re-runnable.** Guard `create_species_from_traps` against
   re-appending; reset `facet_meshtags`/`volume_meshtags` in
   `define_meshtags_and_measures`; split the export setup so it can be redone.
2. **`reinitialise_exports()`.** Close and reopen the field writers against the new
   `Function` objects without truncating, and explicitly clear `Profile1DExport._dofs`,
   `export.D`/`D_expr`, `export.volume_meshtags`, `export.facet_meshtags` and
   `CustomQuantity.ufl_expr`. This is the largest single piece of work, and the prototype
   deliberately avoids it by writing one file per generation.
3. **Regenerate `Mesh1D.vertices` from geometry**, so a mesh whose vertices came from
   adaptation stays self-consistent with `check_borders`.
4. **Make meshtag provenance explicit.** Today it is an implicit `is None` check. AMR
   needs the problem to know whether tags are re-derivable or must be transferred —
   something like `Mesh.retag(new_mesh, parent_maps) -> (ft, ct)`, implemented by
   locators in `Mesh`/`Mesh1D` and by `transfer_meshtag` in `MeshFromXDMF` and in a
   wrapper for user-supplied tags. Refusing to adapt with a clear message is an
   acceptable first answer for the combinations that cannot work (1D + user-supplied
   facet tags + `BaseMeshRefiner`).

### Open questions

- **Conservation** under coarsening (§4). Worth deciding whether FESTIM should offer a
  conservative transfer at all, or document the drift and expose `coarsen_below`.
- **Parallel load balance.** `refine` keeps children on the parent rank, so a moving
  front progressively unbalances the decomposition. Repartitioning costs a full
  redistribution; the prototype is serial and does not address this.
- **Interaction with the adaptive stepsize.** Halving `dt` after a remesh turned out to be
  actively harmful — it cost 80% more timesteps (262 vs 145) and bought nothing, because
  the stepsize controller then spent several steps climbing back. The transferred state is
  a good enough Newton guess to resume at full `dt`. Stiffer problems may differ, so
  `dt_factor` is kept as a knob.
- **`Stepsize.milestones`** and export times: a remesh between two milestones is invisible
  to the current export machinery, which is fine now and will not be once exports survive
  a remesh.
- **Whether a residual estimator is worth it** for reaction-dominated fronts. The gradient
  indicator finds the front, which is all that matters here; a residual estimator would
  cost a `dS` assembly for a marking decision that is not close.

---

## 6. Files

| file | what |
|---|---|
| `indicators.py` | `cell_indicator` (residual-flavoured), `variation_indicator` (absolute), `problem_indicator` (per-subdomain, scattered to parent cells), `dorfler_mark`, `threshold_mark` |
| `adapters.py` | `Mesh1DAdapter`, `BaseMeshRefiner`, `uniform_mesh_1d`, `n_cells` |
| `restart.py` | `rebuild`, `harvest`, `seed_current_from_previous`, `gather_profile`, `total_inventory` |
| `demo_steady.py` | steady AMR loop + uniform-refinement comparison |
| `demo_transient_trapping_1d.py` | the 1D motivating case, `--mode amr\|fine\|coarse\|all` |
| `demo_2d_front.py` | 2D curved front, `--compare` / `--gif`; uses `BaseMeshRefiner` |
| `test_transfer.py` | `pytest amr_prototype/test_transfer.py` — transfer exactness, drift bound, round trip, protected vertices, nD coarsening, indicator cost |
