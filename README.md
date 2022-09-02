# FESTIM

[![CircleCI](https://circleci.com/gh/RemDelaporteMathurin/FESTIM.svg?style=svg&circle-token=ecc5a4a8c75955af6c238d255465bc04dfaaaf8e)](https://circleci.com/gh/RemDelaporteMathurin/FESTIM)
[![codecov](https://codecov.io/gh/RemDelaporteMathurin/FESTIM/branch/master/graph/badge.svg?token=AK3A9CV2D3)](https://codecov.io/gh/RemDelaporteMathurin/FESTIM)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
![GitHub stars](https://img.shields.io/github/stars/RemDelaporteMathurin/FESTIM.svg?logo=github&label=Stars&logoColor=white)
![GitHub forks](https://img.shields.io/github/forks/RemDelaporteMathurin/FESTIM.svg?logo=github&label=Forks&logoColor=white)

FESTIM (Finite Elements Simulation of Tritium in Materials) is a tool for modeling hydrogen transport in materials. 
It simulates the diffusion and trapping of hydrogen, coupled to heat transfer with [FEniCS](https://fenicsproject.org).

:point_right: [Documentation](https://festim.readthedocs.io/)

:point_right: [Examples](https://github.com/RemDelaporteMathurin/FESTIM/tree/main/demos)

## Installation

FESTIM can be installed via pip

    pip install FESTIM

FESTIM requires FEniCS to run.
The FEniCS project provides a [Docker image](https://hub.docker.com/r/fenicsproject/stable/) with FEniCS and its dependencies (python3, UFL, DOLFIN, numpy, sympy...)  already installed. See their ["FEniCS in Docker" manual](https://fenics.readthedocs.io/projects/containers/en/latest/).

For Windows users:

    docker run -ti -v ${PWD}:/home/fenics/shared --name fenics quay.io/fenicsproject/stable:latest

For Linux users:

    docker run -ti -v $(pwd):/home/fenics/shared --name fenics quay.io/fenicsproject/stable:latest

Run the tests:

    pytest-3 test/

## Visualisation
FESTIM results are exported to .csv, .txt or XDMF. The latter can then be opened in visualisation tools like [ParaView](https://www.paraview.org/) or [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/).

## References
- R. Delaporte-Mathurin, _et al._, _Finite Element Analysis of Hydrogen Retention in ITER Plasma Facing Components using FESTIM_. Nuclear Materials and Energy 21: 100709. DOI: [10.1016/j.nme.2019.100709](https://doi.org/10.1016/j.nme.2019.100709).

- R. Delaporte-Mathurin, _et al._, _Parametric Optimisation Based on TDS Experiments for Rapid and Efficient Identification of Hydrogen Transport Materials Properties_. Nuclear Materials and Energy, 26 mars 2021, 100984. https://doi.org/10.1016/j.nme.2021.100984.

- R. Delaporte-Mathurin, _et al._, _Parametric Study of Hydrogenic Inventory in the ITER Divertor Based on Machine Learning_. Scientific Reports 10: 17798. https://doi.org/10.1038/s41598-020-74844-w.

- R. Delaporte-Mathurin, _et al._, _Influence of Interface Conditions on Hydrogen Transport Studies_. Nuclear Fusion 61 (2021): 036038. https://doi.org/10.1088/1741-4326/abd95f.

- R. Delaporte-Mathurin, _et al._, _Fuel Retention in WEST and ITER Divertors Based on FESTIM Monoblock Simulations_. Nuclear Fusion 61, nᵒ 12: 126001. https://doi.org/10.1088/1741-4326/ac2bbd.

- E. A. Hodille _et al._, _Modelling of Hydrogen Isotopes Trapping, Diffusion and Permeation in Divertor Monoblocks under ITER-like Conditions_. Nuclear Fusion, 2021. https://doi.org/10.1088/1741-4326/ac2abc.
