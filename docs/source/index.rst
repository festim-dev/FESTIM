.. FESTIM documentation master file, created by
   sphinx-quickstart on Thu Jul 28 11:09:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FESTIM
======

FESTIM is a FEniCS-based application for solving coupled hydrogen transport - heat transfer simulations.
The tool is based on the finite element methods to solve the McNabb & Foster transport equations and the heat equation.
It is capable of solving 1D/2D/3D multimaterial simulations and provides support for a wide-range of boundary conditions, which makes it a very versatile tool that can be adapted to many use cases.
Moreover, users can rapidly get started with FESTIM thanks to its python API.

FESTIM was originally developed at the `Institute for Magnetic Fusion Research (IRFM) <https://irfm.cea.fr/en/index.php>`_ and the `Process and Materials Sciences Laboratory (LSPM) <https://www.lspm.cnrs.fr/en/home/>`_.


.. admonition:: Recommended publication for citing
   :class: tip

   Rémi Delaporte-Mathurin, Etienne A. Hodille, Jonathan Mougenot, Yann Charles, and Christian Grisolia.
   "`Finite Element Analysis of Hydrogen Retention in ITER Plasma Facing Components Using FESTIM.
   Nuclear Materials and Energy 21 (2019): 100709. <https://doi.org/10.1016/j.nme.2019.100709>`_",

--------
Contents
--------

.. toctree::
   :maxdepth: 1

   getting_started
   examples
   theory
   userguide/index
   devguide/index
   api/festim
   publications


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
