============
Installation
============


Because the main dependency FEniCSx (dolfinx) cannot be installed natively on Windows, Windows users must use the `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_ (WSL). FEniCSx can be installed natively on MacOs and Linux.

.. tip::
    You can install `WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_ by running
    :code:`wsl --install` on a command prompt window.
    To launch into WSL, simply enter :code:`wsl`.

    `Visual Studio Code <https://code.visualstudio.com/>`_ is the recommended IDE to 
    use with Windows due to the 
    `WSL extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl>`_.


Option 1: Conda (Recommended)
-----------------------------

The easiest way to install FESTIM with all its dependencies (including dolfinx) is via Conda.

First install a conda distribution, e.g.
`Anaconda <https://docs.anaconda.com/anaconda/install/>`_ or
`Miniforge <https://github.com/conda-forge/miniforge>`_.

.. tip::

    You can install Miniforge on Linux and MacOS by entering::

        curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        bash Miniforge3-$(uname)-$(uname -m).sh

    See the `Miniforge <https://github.com/conda-forge/miniforge#install>`_ and
    `Anaconda <https://docs.anaconda.com/anaconda/install/>`_ documentation for other
    installation methods.

Then run the following commands to create an environment and install FESTIM::

    conda create -n festim-env
    conda activate festim-env       
    conda install -c conda-forge festim

**Specific versions**

To install a specific version of FESTIM, you can specify it in the installation command::

    conda install -c conda-forge festim=0.9.0

To check which version of FESTIM is currently installed, run::

    conda list festim

**Nightly / development version**

We provide a nightly conda build via our own anaconda channel. You can install it with::

    # Nightly / development version
    conda install -c festim-dev/label/nightly -c conda-forge festim

    # Or with a pin to latest nightly
    conda install -c festim-dev/label/nightly -c conda-forge festim=*=nightly_*


Option 2: Docker
----------------

Installing the FEniCSx docker container and then installing FESTIM with pip inside it is also an option::

    docker run -ti dolfinx/dolfinx:stable
    pip install festim


Option 3: From Source
---------------------

When none of the above methods are possible, users can build FEniCSx from source and then install FESTIM and all dependencies manually. Please refer to the `FEniCSx documentation <https://docs.fenicsproject.org/>`_ for complete instructions on building dolfinx from source.
