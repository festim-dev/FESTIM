============
Installation
============

FESTIM is installed in two steps:
    1. `Installing FEniCSx`_ through Anaconda or Docker
    2. `Installing FESTIM`_ through pip in the FEniCS environment

Installing FEniCSx
-----------------

FEniCSx can be installed with Anaconda on MacOs and Linux. 
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
    conda install -c conda-forge fenics-dolfinx tqdm scifem


Installing FESTIM
-----------------

This version of FESTIM is under development and is not yet available on PyPI.
FESTIM can then be installed using pip::

    pip install git+https://github.com/FESTIM-dev/FESTIM@fenicsx
