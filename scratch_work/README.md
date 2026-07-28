# Scratch Work

This directory contains temporary and exploratory scripts related to development of co-dimensional meshes

## Files

- `petsc_solver_for_manifold_derivatives.py` — exploratory work around PETSc solver behavior for manifold derivative problems, authored by [Jorgen S. Dokken](https://github.com/jorgensd).
- `interface_trapping.py` — prototype script of trapping at an interface (without diffusion), written by [Remi Delaporte-Mathurin](https://github.com/RemDelaporteMathurin).
- `interface_trapping_custom_solver.py` — variant of the above using the modified PETSc solver setup, with diffusion implemented along the interface. 
- `advection_mwe.py` — minimal working example script taken from [Remi's discourse post](https://fenicsproject.discourse.group/t/coupled-problem-with-codim-1-submesh-wrong-derivative/19688).
- `advection_mwe_custom_solver.py` — version of the above MWE that uses the custom PETSc solver. 

## Notes

These files are not part of the main package and may be used for debugging, experiments, or development investigations.
