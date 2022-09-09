===============
Getting started
===============

Install
*******

Installing FEniCS
-----------------

FESTIM requires FEniCS to run.

It can be installed using `Docker <https://www.docker.com/>`_::

    docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable:latest

.. note::
    :code:`$(pwd)` needs to be replaced by :code:`${PWD}` on Windows


Alternatively, FEniCS can be installed using `Conda <https://docs.continuum.io/anaconda/install/>`_::

    conda create -n fenicsproject -c conda-forge fenics
    source activate fenicsproject

For more information on how to install FEniCS, see `Download <https://fenicsproject.org/download/archive/>`_ on the FEniCS website.


Installing FESTIM
-----------------

FESTIM can then be installed using pip::

    pip install FESTIM


Tests
*****

Check that everything is working properly by running the tests::

    pytest test
