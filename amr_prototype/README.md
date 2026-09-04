# AMR prototype

Investigation into adaptive mesh refinement for FESTIM. **Nothing in `src/festim/` is
modified** — everything here drives FESTIM from the outside.

Read [`AMR_DESIGN.md`](AMR_DESIGN.md) for the findings and the proposed API.

```bash
cd amr_prototype

python demo_steady.py                                  # steady AMR + uniform comparison
python demo_transient_trapping_1d.py --mode all        # 1D motivating case, ~15 s
python demo_2d_front.py                                # 2D curved front + figure, ~1 min
python demo_2d_front.py --gif                          # animation as well
python demo_2d_front.py --compare                      # + uniform runs (slow)
python -m pytest test_transfer.py                      # sanity checks

# quicker
python demo_transient_trapping_1d.py --mode all --final-time 1e4
```

Run from inside `amr_prototype/` — the modules import each other flatly. Output goes to
`out/`.

## Headline results

- **Steady (1D):** 50 adaptively refined cells reach the 10240-cell uniform answer to
  0.004%.
- **Transient (1D):** AMR removes the 15% overshoot above trap saturation that a uniform
  100-cell mesh produces, using ≤943 cells against 2000 — but is *slower*, because a 1D
  solve costs only 20% more for 20× the cells. The 1D win is correctness, not speed.
- **Transient (2D):** at the same front resolution, **4.6× fewer cells and 3.2× faster**
  than the equivalent uniform mesh (24 970 cells / 50 s vs 115 200 cells / 163 s), with
  a lower overshoot. This is where AMR pays for itself.
- **DOLFINx gotcha:** `refine`'s `parent_cell` is not a valid index into the refined mesh
  (cells are reordered). Carry per-cell data with `transfer_meshtag` instead.
- **Silent-failure gotcha:** an absolute SNES tolerance is compared against a residual
  carrying a factor of the cell measure, so `atol` that works in 1D can be larger than
  the entire residual in 2D — and every refinement makes it more permissive.
