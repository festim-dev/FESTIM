============
Installation
============

Installing FEniCS
-----------------

FESTIM requires FEniCS to run.

FEniCS can be installed with Anaconda on MacOs and Linux. 
In order to use the Anaconda distribution on Windows, use the `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_. 
First `install Anaconda <https://docs.continuum.io/anaconda/install>`_ then run the following commands::

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
