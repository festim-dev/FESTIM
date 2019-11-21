# FESTIM

[![CircleCI](https://circleci.com/gh/RemiTheWarrior/FESTIM.svg?style=svg)](https://circleci.com/gh/RemiTheWarrior/FESTIM) 
[![PEP8](https://img.shields.io/badge/code%20style-pep8-violet.svg)](https://www.python.org/dev/peps/pep-0008/)
![GitHub stars](https://img.shields.io/github/stars/RemiTheWarrior/FESTIM.svg?logo=github&label=Stars&logoColor=white)
![GitHub forks](https://img.shields.io/github/forks/RemiTheWarrior/FESTIM.svg?logo=github&label=Forks&logoColor=white)

FESTIM (Finite Elements Simulation of Tritium in Materials) is a tool for modeling Hydrogen Isotopes (HIs) transport in materials. 
The governing equations are composed of:
- **Fick's law** of diffusion of species based on concentration gradient
- **Soret** effect
- Trapping/detrapping **macroscopic rate equations**
- **Heat equation**


The following features are included in this tool:
- Mesh import from XDMF files
- **Adaptive stepsize**
- **Temperature** is defined by:
    - user-defined expression
    - solving transient/stationnary heat equation
- Multiple intrinsic/extrinsic traps with **non-homogeneous density distribution**
- Wide range of built-in boundary conditions (Sievert's law, recombination flux, experimental data, user-defined expression...)
- **Derived quantities** computation (surface fluxes, volume integrations, extrema over domains, mean values over domains...)

FESTIM spatially discretises the PDEs using the Finite Element Methods. At this extent, we chose to use the C++/Python library [FEniCS](https://fenicsproject.org). 
The implicit time discretisation method is backward Euler.
PDEs are solved using FEniCS' Newton nonlinear solver. A library of generic functions is provided so that users can run custom simulations in addition to the flexibility of [FEniCS](https://fenicsproject.org) built-in functions.

## Run FESTIM in Docker (e.g. on Windows, Mac, many Linux distributions)
The FEniCS project provides a [Docker image](https://hub.docker.com/r/fenicsproject/stable/) with FEniCS and its dependencies (python3, UFL, DOLFIN, numpy, sympy...)  already installed. See their ["FEniCS in Docker" manual](https://fenics.readthedocs.io/projects/containers/en/latest/).

Get Docker [here](https://www.docker.com/community-edition).

Pull the Docker image and run the container, sharing a folder between the host and container:

    docker run -ti -v $(pwd):/home/fenics/shared --name fenics quay.io/fenicsproject/stable:latest

Clone FESTIM's git repository:

    git clone https://github.com/RemiTheWarrior/FESTIM
    
Run the tests:

    pytest-3 Tests/
## Visualisation
FESTIM allows users to export their data to .csv, .txt or to a XDMF format with an XML interface. The latter can then be opened in visualisation tools like [ParaView](https://www.paraview.org/) or [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/).
<p align="center">
  <img alt="performance" src="https://user-images.githubusercontent.com/40028739/69346147-9abb6980-0c72-11ea-80e7-9c0a76659268.png" width="40%"> <img alt="performance" src="https://user-images.githubusercontent.com/40028739/69346752-9d6a8e80-0c73-11ea-96c1-27b6104eb9ff.png" width="40%">
</p>

## References
- R. Delaporte-Mathurin, et al., ‘Finite Element Analysis of Hydrogen Retention in ITER Plasma Facing Components Using FESTIM’. Nuclear Materials and Energy 21: 100709. https://doi.org/10.1016/j.nme.2019.100709.

