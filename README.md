# FESTIM

[![CircleCI](https://circleci.com/gh/RemiTheWarrior/FESTIM.svg?style=svg)](https://circleci.com/gh/RemiTheWarrior/FESTIM) 
[![PEP8](https://img.shields.io/badge/code%20style-pep8-violet.svg)](https://www.python.org/dev/peps/pep-0008/)
![GitHub stars](https://img.shields.io/github/stars/RemiTheWarrior/FESTIM.svg?logo=github&label=Stars&logoColor=white)
![GitHub forks](https://img.shields.io/github/forks/RemiTheWarrior/FESTIM.svg?logo=github&label=Forks&logoColor=white)

FESTIM (Finite Elements Simulation of Tritium in Materials) is a tool for modeling hydrogen isotopes (HIs) behavior in materials. 
The governing equations are composed of:
- Fick's law of diffusion of species based on concentration gradient
- Trapping/detrapping macroscopic rate equations
- Heat equation for temperature field

The following features are included in this tool:
- 1D transient simulations of HIs diffusion/trapping/detrapping in multimaterial domains
- 1D transient/stationary simulation of heat diffusion
- Analytical temperature expression
- Extrinsic traps
- Non-homogeneous trap distribution
- Nonzero initial values
- Boundary conditions: Dirichlet, Pressure/solubility

FESTIM spatially discretizes the PDEs using the Finite Element Methods. At this extent, we chose to use the C++/Python library [FEniCS](https://fenicsproject.org). The implicit time discretisation method is backward Euler and the PDEs are solved using FEniCS' Newton nonlinear solver. We also provides a library of generic functions so that users can write custom simulations in addition to the flexibility of [FEniCS](https://fenicsproject.org) built-in functions.

## Run FESTIM in Docker (e.g. on Windows, Mac, many Linux distributions)
The FEniCS project provides a [Docker image](https://hub.docker.com/r/fenicsproject/stable/) with FEniCS and its dependencies (python3, UFL, DOLFIN, numpy, sympy...)  already installed. See their ["FEniCS in Docker" manual](https://fenics.readthedocs.io/projects/containers/en/latest/).

Get Docker [here](https://www.docker.com/community-edition).

Pull the Docker image and run the container, sharing a folder between the host and container:

    docker run -ti -v $(pwd):/home/fenics/shared --name fenics quay.io/fenicsproject/stable:latest

Clone FESTIM's git repository:

    git clone https://github.com/RemiTheWarrior/FESTIM
    
Run the tests:

    pytest-3 Main/test.py
## Visualisation
FESTIM allows users to export their data to .csv, .txt or to a XDMF format with an XML interface. The latter can then be opened in visualisation tools like [ParaView](https://www.paraview.org/) or [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/).

## Todo
- write more tests
- write more demos
## References
- Hodille _et al._ _Macroscopic rate equation modeling of trapping/detrapping of hydrogen isotopes in tungsten materials_. JNM 467 (2015) 424-431
