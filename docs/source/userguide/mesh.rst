====
Mesh
====

Meshes are required to discretise the geometrical domain of the simulation.
As FESTIM is not a meshing tool, the meshing capabilities are limited to simple 1D meshes.

---------
1D meshes
---------

The easiest way to define a 1D mesh in FESTIM is to define it from a list of vertices:

.. code-block:: python

    mesh = MeshFromVertices([0, 1, 2, 4, 5, 10])

For bigger meshes, use the numpy library to generate an array of vertices.

.. code-block:: python

    import numpy as np

    mesh = MeshFromVertices(np.linspace(0, 10, num=1000))

Numpy arrays can be combined to have local refinements:

.. code-block:: python

    import numpy as np

    vertices = np.concatenate(
        [
            np.linspace(0, 1e-6, num=100),  # 99 cells between 0 and 1 micron
            np.linspace(1e-6, 1e-4, num=100),  # 99 cells between 1 micron and 0.1 mm
            np.linspace(1e-4, 1e-2, num=10)  # 9 cells between 0.1 mm and 1 cm
        ]
    )
    mesh = MeshFromVertices(vertices)

----------------
Meshes from XDMF
----------------

More complex meshes can be read from XDMF files:

.. code-block:: python

    mesh = MeshFromXDMF(volume_file="volume_file.xdmf", boundary_file="boundary_file.xdmf")

The recommended workflow is to mesh your geometry with your favourite meshing software (`SALOME <https://www.salome-platform.org/?lang=fr>`_, `gmsh <https://gmsh.info/>`_...) and convert the produced mesh with `meshio <https://github.com/nschloe/meshio>`_.

GMSH example
------------

The DOLFINx tutorial gives an `example <https://jorgensd.github.io/dolfinx-tutorial/chapter1/membrane_code.html#creating-the-mesh>`_ of mesh generation with gmsh.

------------------
meshes from FEniCS
------------------

See the `FEniCS documentation <https://fenicsproject.org/olddocs/dolfin/latest/python/demos/built-in-meshes/demo_built-in-meshes.py.html>`_ for more built-in meshes.
