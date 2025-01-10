============
Installation
============

FESTIM is installed in two steps:
    1. `Installing FEniCS`_ through Anaconda or Docker
    2. `Installing FESTIM`_ through pip on the FEniCS environment

Installing FEniCS
-----------------

FEniCS can be installed with Anaconda on MacOs and Linux. 
In order to use the Anaconda distribution on Windows, 
use the `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_. 

.. tip::
    You can install `WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_ by running
    :code:`wsl --install` on a command prompt window.
    To launch into WSL, simply enter :code:`wsl`.

    `Visual Studio Code <https://code.visualstudio.com/>`_ is the recommended IDE to 
    use with Windows due to the 
    `WSL extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl>`_.

First `install Anaconda <https://docs.continuum.io/anaconda/install>`_,

.. tip::

    You can install Anaconda on most Linux distributions by entering::

        curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
        bash ./Anaconda3-2024.06-1-Linux-x86_64.sh

    You can install other versions by replacing :code:`Anaconda3-2024.06-1-Linux-x86_64.sh` 
    with another from `the official repository <https://repo.anaconda.com/archive/>`_.

then run the following commands::

    conda create -n festim-env
    conda activate festim-env       
    conda install -c conda-forge fenics numpy=1.24

Alternatively, It can be installed using `Docker <https://www.docker.com/>`_::

    docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable:latest

.. note::
    :code:`$(pwd)` needs to be replaced by :code:`${PWD}` on Windows


For more information on how to install FEniCS, see `Download <https://fenicsproject.org/download/archive/>`_ on the FEniCS website.


Installing FESTIM
-----------------

FESTIM can then be installed using pip::

    pip install FESTIM

Specific versions of FESTIM can be installed with::

    pip install FESTIM==[version]

with the desired version tag.  For example::

    pip install FESTIM==0.9

To upgrade FESTIM to the latest version, run::

    pip install --upgrade FESTIM

To uninstall FESTIM, run::

    pip uninstall FESTIM


Installing the ``fenicsx`` branch version
-------------------------------------------

This version of FESTIM is under development and is not yet available on PyPI.
It runs on ``dolfinx`` instead of ``fenics`` and can be installed on Linux and MacOS.
If you are on Windows, you can use the Windows Subsystem for Linux (simply follow the instructions above).

Create a conda environment with ``dolfinx``::

    conda create -n festim-env
    conda activate festim-env       
    conda install -c conda-forge fenics-dolfinx

Install the correct FESTIM version with::

    pip install git+https://github.com/FESTIM-dev/FESTIM@fenicsx
