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

FESTIM requires FEniCS to run.
To install a FEniCS container consult the [Getting started](https://festim.readthedocs.io/en/latest/getting_started.html) section in the docs.

Once inside a FEniCS container environement, FESTIM can be installed via pip:

    pip install FESTIM


## Visualisation
FESTIM results are exported to .csv, .txt or XDMF. The latter can then be opened in visualisation tools like [ParaView](https://www.paraview.org/) or [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/).

## References
- R. Delaporte-Mathurin, _et al._, _Finite Element Analysis of Hydrogen Retention in ITER Plasma Facing Components using FESTIM_. Nuclear Materials and Energy, 21, (2019). https://doi.org/10.1016/j.nme.2019.100709.

- R. Delaporte-Mathurin, _et al._, _Parametric Study of Hydrogenic Inventory in the ITER Divertor Based on Machine Learning_. Scientific Reports, 10, (2020). https://doi.org/10.1038/s41598-020-74844-w.

- J. Dark, _et al._, _Influence of hydrogen trapping on WCLL breeding blanket performances_. Nuclear Fusion, 62, (2021). https://doi.org/10.1088/1741-4326/ac28b0.

For full list of publications using FESTIM see the [publications](https://festim.readthedocs.io/en/latest/publications.html) section in the docs