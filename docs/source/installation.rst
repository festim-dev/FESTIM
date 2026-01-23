=======
Install
=======

Operating System Requirements
-----------------------------
FESTIM's compatibility is dependent on FEniCS, which supports:

- **Linux**: Fully supported
- **macOS**: Fully supported
- **Windows**: Not directly supported, but available through:

  - Windows Subsystem for Linux (WSL) (recommended approach)
  - Docker containers

.. note::
    If you're using Windows, please follow the WSL installation instructions below or use the Docker approach.

Windows Subsystem for Linux
----------------------------
In order to use the conda distribution on Windows, 
use the `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_. 

.. tip::
    You can install `WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_ by running
    :code:`wsl --install` on a command prompt window.
    To launch into WSL, simply enter :code:`wsl`.

    `Visual Studio Code <https://code.visualstudio.com/>`_ is the recommended IDE to 
    use with Windows due to the 
    `WSL extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl>`_.


Installing FESTIM with Conda
----------------------------

First `install Anaconda <https://docs.continuum.io/anaconda/install>`_,

.. tip::

    You can install Anaconda on most Linux distributions by entering::

        curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
        bash ./Anaconda3-2024.06-1-Linux-x86_64.sh

    You can install other versions by replacing :code:`Anaconda3-2024.06-1-Linux-x86_64.sh` 
    with another from `the official repository <https://repo.anaconda.com/archive/>`_.

FESTIM can be installed with Anaconda on MacOs and Linux::

    conda create -n festim-env
    conda activate festim-env       
    conda install -c conda-forge festim==1.4

.. tip::

    You can install the latest version of FESTIM by running::

        conda install -c conda-forge festim

    To install a specific version, run::

        conda install -c conda-forge festim==[version]

    with the desired version tag.  For example::

        conda install -c conda-forge festim==0.9

    To upgrade FESTIM to the latest version, run::

        conda install -c conda-forge festim --update-deps

    To uninstall FESTIM, run::

        conda uninstall festim


Installing FESTIM with pip
--------------------------

.. note::
    FEniCS is required for the pip install to work. Consider installing it either with Conda, Docker, or from source.

FESTIM can be installed using pip::

    pip install FESTIM

Specific versions of FESTIM can be installed with::

    pip install FESTIM==[version]

with the desired version tag.  For example::

    pip install FESTIM==0.9

To upgrade FESTIM to the latest version, run::

    pip install --upgrade FESTIM

To uninstall FESTIM, run::

    pip uninstall FESTIM

Using Docker
------------

Alternatively, FESTIM can be installed using `Docker <https://www.docker.com/>`_.

First create a docker container based on the FEniCS docker image::

    docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable:latest

.. note::
    :code:`$(pwd)` needs to be replaced by :code:`${PWD}` on Windows

.. note::
    For more information on how to install FEniCS, see `Download <https://fenicsproject.org/download/archive/>`_ on the FEniCS website.

Then install FESTIM using pip (see the `Installing FESTIM with pip`_ section above).


Installing the ``2.0-alpha`` version
------------------------------------

This version of FESTIM is not production-ready but available as an alpha version.
This version is developed on the ``fenicsx`` branch of the FESTIM repository.
It runs on ``dolfinx`` instead of ``fenics`` and can be installed on Linux and MacOS.
If you are on Windows, you can use the Windows Subsystem for Linux (simply follow the instructions above).

Install it with Conda::

    conda create -n festim-env
    conda activate festim-env       
    conda install -c conda-forge festim==2.0a4

