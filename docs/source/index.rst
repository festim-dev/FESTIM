FESTIM
======

FESTIM is a FEniCS-based application for solving coupled hydrogen transport - heat transfer simulations.
The tool is based on the finite element methods to solve the McNabb & Foster transport equations and the heat equation.
It is capable of solving 1D/2D/3D multimaterial simulations and provides support for a wide-range of boundary conditions, which makes it a very versatile tool that can be adapted to many use cases.
Moreover, users can rapidly get started with FESTIM thanks to its python API.

.. All the logos are from this collection https://www.svgrepo.com/collection/solar-linear-icons/

.. grid:: 3
      :gutter: 2

      .. grid-item::

         .. card:: Installation
            :img-top: images/icons/running-2-svgrepo-com.svg
            :link: installation
            :link-type: doc

      .. grid-item::

         .. card:: User guide
            :img-top: images/icons/book-2-svgrepo-com.svg
            :link: userguide/index
            :link-type: doc

      .. grid-item::

         .. card:: Tutorials
            :img-top: images/icons/clapperboard-play-svgrepo-com.svg
            :link: https://github.com/festim-dev/FESTIM-workshop

      .. grid-item::

         .. card:: Developer guide
            :img-top: images/icons/code-square-svgrepo-com.svg
            :link: devguide/index
            :link-type: doc

      .. grid-item::
         .. card:: V&V
            :img-top: images/icons/check-square-svgrepo-com.svg
            :link: https://festim-vv-report.readthedocs.io/en/latest/

      .. grid-item::
         .. card:: API reference
            :img-top: images/icons/keyboard-svgrepo-com.svg
            :link: api/index
            :link-type: doc

FESTIM was originally developed at the `Institute for Magnetic Fusion Research (IRFM) <https://irfm.cea.fr/en/index.php>`_ and the `Process and Materials Sciences Laboratory (LSPM) <https://www.lspm.cnrs.fr/en/home/>`_.
Various research institutions and private companies now contribute actively to FESTIM's development.
For more information, feel free to ask questions on the `FESTIM Discourse Page <https://festim.discourse.group/>`_.


.. admonition:: Recommended publication for citing
   :class: tip

   Rémi Delaporte-Mathurin, James Dark, Gabriele Ferrero, Etienne A. Hodille, Vladimir Kulagin, Samuele Meschini,
   "`FESTIM: An open-source code for hydrogen transport simulations.
   International Journal of Hydrogen Energy 63 (2024): 786-802. <https://doi.org/10.1016/j.ijhydene.2024.03.184>`_",

Map of FESTIM users
-------------------

.. raw:: html

   <iframe src="_static/map.html" width="800" height="600"></iframe>


.. admonition:: Add your institution

   If you would like your institution to be added to this map, please `open an issue <https://github.com/festim-dev/FESTIM/issues/new/choose>`_.


--------
Contents
--------

.. toctree::
   :maxdepth: 1

   installation
   examples
   theory
   userguide/index
   devguide/index
   api/index
   publications