===============
Getting started
===============

Install
*******

Installing FEniCS
-----------------

FESTIM requires FEniCS to run.

The FEniCS project provides a prebuilt Anaconda python package (Linux and MacOS only) with FEniCS and its dependencies (python3, UFL, DOLFIN, numpy, sympy...)  already installed. 
Anaconda can also be used in Windows using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) with your favourite linux distro. 
First [install Anaconda](https://docs.continuum.io/anaconda/install) then run the following commands 

    conda create -n festim-env -c conda-forge fenics
    source activate festim-env

Alternatively, It can be installed using `Docker <https://www.docker.com/>`_::

    docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable:latest

.. note::
    :code:`$(pwd)` needs to be replaced by :code:`${PWD}` on Windows


For more information on how to install FEniCS, see `Download <https://fenicsproject.org/download/archive/>`_ on the FEniCS website.


Installing FESTIM
-----------------

FESTIM can then be installed using pip::

    pip install FESTIM


Tests
*****

Check that everything is working properly by running the tests::

    pytest test
