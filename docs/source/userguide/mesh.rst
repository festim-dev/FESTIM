.. _mesh_guide:

====
Mesh
====

.. testsetup:: mesh

    import festim as F

Meshes are required to discretise the geometrical domain of the simulation.
As FESTIM is not a meshing tool, its meshing capabilities are limited to simple 1D meshes.
Higher-dimensional meshes are built with `DOLFINx <https://docs.fenicsproject.org/dolfinx/main/python/>`_ or with external meshing software (such as `gmsh <https://gmsh.info/>`_ or `SALOME <https://www.salome-platform.org/>`_) and read into FESTIM.

Regardless of how it is defined, the mesh is passed to the :code:`mesh` attribute of the problem.

-----------------
Tags and locators
-----------------

Every :class:`festim.VolumeSubdomain` and :class:`festim.SurfaceSubdomain` carries an integer :code:`id`, and FESTIM has to work out which cells and which facets it refers to. There are two mechanisms:

- **meshtags** — integer arrays attached to the cells and facets of the mesh, normally written by the meshing software as *physical groups* (gmsh) or *mesh groups* (SALOME). The subdomain :code:`id` is matched against the tag value.
- **locators** — a Python function of the coordinates, evaluated by FESTIM to find the entities itself. This is what is used when the mesh carries no tags.

Meshes built by FESTIM tag themselves. Meshes coming from an external mesher normally bring their own tags. Meshes built with the DOLFINx helpers carry none and need locators.

.. note::
    Cell (volume) tags and facet (surface) tags are stored in two separate meshtags
    objects, so a volume and a surface may share the same id without ambiguity.
    :ref:`Codimensional (manifold) Subdomains` are the exception: they are stored in the
    facet meshtags, so their ids must be unique among the surfaces.

---------
1D meshes
---------

The easiest way to define a 1D mesh in FESTIM is to define it from a list of vertices (see :class:`festim.Mesh1D`):

.. testcode:: mesh

    mesh = F.Mesh1D(vertices=[0, 1, 2, 4, 5, 10])

For bigger meshes, use the numpy library to generate an array of vertices.

.. testcode:: mesh

    import numpy as np

    mesh = F.Mesh1D(vertices=np.linspace(0, 10, num=1000))

Numpy arrays can be combined to have local refinements:

.. testcode:: mesh

    import numpy as np

    vertices = np.concatenate(
        [
            np.linspace(0, 1e-6, num=100),  # 99 cells between 0 and 1 micron
            np.linspace(1e-6, 1e-4, num=100),  # 99 cells between 1 micron and 0.1 mm
            np.linspace(1e-4, 1e-2, num=10)  # 9 cells between 0.1 mm and 1 cm
        ]
    )
    mesh = F.Mesh1D(vertices=vertices)

Duplicated vertices (here at 1e-6 and 1e-4) are removed automatically.

Several disconnected solids can be represented with one mesh by giving the vertices as a list of lists, one per solid.
No cell is created between the last vertex of a block and the first vertex of the next one, leaving a gap which can then be coupled through an enclosure:

.. testcode:: mesh

    mesh = F.Mesh1D(vertices=[[0, 0.1, 0.2, 0.3], [1, 1.1, 1.2]])

Volume subdomains of 1D meshes are defined with :class:`festim.VolumeSubdomain1D` and surfaces with :class:`festim.SurfaceSubdomain1D` (see :doc:`Subdomains & Materials <subdomains>`).

-------------------
Meshes from DOLFINx
-------------------

Any :code:`dolfinx.mesh.Mesh` object can be used in FESTIM by wrapping it in :class:`festim.Mesh`.
This is the simplest way to obtain 2D and 3D meshes of simple geometries, using the `built-in meshes of DOLFINx <https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.mesh.html>`_:

.. testcode:: mesh

    from mpi4py import MPI
    from dolfinx.mesh import create_unit_square

    dolfinx_mesh = create_unit_square(MPI.COMM_WORLD, 50, 50)

    mesh = F.Mesh(mesh=dolfinx_mesh)

Similarly, :code:`create_rectangle`, :code:`create_box`, :code:`create_unit_cube`... can be used.

Since such meshes do not carry any tag, the volume and surface subdomains must be located with a ``locator`` function (see :ref:`Surface Subdomains` and :ref:`Volume Subdomains`):

.. testcode:: mesh

    import numpy as np

    my_mat = F.Material(D_0=1, E_D=0.1)

    left_half = F.VolumeSubdomain(
        id=1, material=my_mat, locator=lambda x: x[0] <= 0.5
    )
    right_half = F.VolumeSubdomain(
        id=2, material=my_mat, locator=lambda x: x[0] >= 0.5
    )
    left_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[0], 0))
    right_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 1))

.. note::

    Alternatively, if you already have facet and cell meshtags (for instance from gmsh, see below), they can be given directly to the problem through its :code:`facet_meshtags` and :code:`volume_meshtags` attributes.
    The subdomain ids must then match the values of the meshtags.

Coordinate systems
------------------

By default, the equations are solved in cartesian coordinates.
Cylindrical (1D or 2D meshes) and spherical (1D meshes only) coordinates can be selected with the :code:`coordinate_system` argument:

.. testcode:: mesh

    mesh = F.Mesh1D(vertices=np.linspace(1e-3, 2e-3, num=100), coordinate_system="spherical")

.. testcode:: mesh

    mesh = F.Mesh(mesh=dolfinx_mesh, coordinate_system="cylindrical")

In cylindrical coordinates :code:`x[0]` is the radial coordinate :math:`r` and :code:`x[1]` is :math:`z`; in spherical coordinates :code:`x[0]` is :math:`r`.

----------------
Meshes from XDMF
----------------

More complex meshes can be read from XDMF files (see :class:`festim.MeshFromXDMF`): one file containing the mesh and the volume (cell) tags, and one containing the surface (facet) tags.

.. testsetup:: mesh

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import XDMFFile
    from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags

    _mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    _num_cells = _mesh.topology.index_map(2).size_local
    _ct = meshtags(
        _mesh, 2, np.arange(_num_cells, dtype=np.int32), np.full(_num_cells, 1, dtype=np.int32)
    )
    _facets = locate_entities_boundary(_mesh, 1, lambda x: np.isclose(x[0], 0))
    _ft = meshtags(_mesh, 1, _facets, np.full(len(_facets), 1, dtype=np.int32))
    _mesh.topology.create_connectivity(1, 2)
    _mesh.name = _ct.name = _ft.name = "Grid"
    for _name, _tags in [("volume_mesh.xdmf", _ct), ("surface_mesh.xdmf", _ft)]:
        with XDMFFile(MPI.COMM_WORLD, _name, "w") as _f:
            _f.write_mesh(_mesh)
            _f.write_meshtags(_tags, _mesh.geometry)

.. testcode:: mesh

    mesh = F.MeshFromXDMF(volume_file="volume_mesh.xdmf", facet_file="surface_mesh.xdmf")

.. testcleanup:: mesh

    import os
    for _name in ["volume_mesh", "surface_mesh"]:
        for _ext in [".xdmf", ".h5"]:
            if os.path.exists(_name + _ext):
                os.remove(_name + _ext)

When such a mesh is used, the meshtags are read from the files and the subdomains do not need a ``locator``: their ids simply have to match the tags in the files.

The XDMF files must be readable by DOLFINx.
By default, the mesh and the meshtags are looked up under the name ``"Grid"`` in the files, which is what `meshio <https://github.com/nschloe/meshio>`_ writes.
Different names can be given with the ``mesh_name``, ``volume_meshtags_name`` and ``surface_meshtags_name`` arguments.
For instance, files written by :code:`dolfinx.io.XDMFFile` use the names of the :code:`dolfinx.mesh.Mesh` and :code:`dolfinx.mesh.MeshTags` objects (``"mesh"`` and ``"mesh_tags"`` by default).

The recommended workflow is to mesh your geometry with your favourite meshing software (`SALOME <https://www.salome-platform.org/>`_, `gmsh <https://gmsh.info/>`_...) and either read it directly with DOLFINx (`gmsh`) or convert the produced mesh to XDMF with `meshio`.

For step-by-step tutorials of GMSH and SALOME mesh generation, please see the `tutorials page <https://festim-workshop.readthedocs.io/en/latest/intro.html>`_.

