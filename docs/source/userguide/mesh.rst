====
Mesh
====

.. testsetup::
    
    from festim import MeshFromVertices, MeshFromXDMF

Meshes are required to discretise the geometrical domain of the simulation.
As FESTIM is not a meshing tool, the meshing capabilities are limited to simple 1D meshes.

---------
1D meshes
---------

The easiest way to define a 1D mesh in FESTIM is to define it from a list of vertices (see :class:`festim.MeshFromVertices`):

.. testcode::

    mesh = MeshFromVertices([0, 1, 2, 4, 5, 10])

For bigger meshes, use the numpy library to generate an array of vertices.

.. testcode::

    import numpy as np

    mesh = MeshFromVertices(np.linspace(0, 10, num=1000))

Numpy arrays can be combined to have local refinements:

.. testcode::

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

More complex meshes can be read from XDMF files (see :class:`festim.MeshFromXDMF`):

.. testsetup::

    import fenics as f

    mesh = f.UnitSquareMesh(10, 10)

    volume_markers = f.MeshFunction("size_t", mesh, mesh.topology().dim(), 1)
    surface_markers = f.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 1)

    f.XDMFFile("volume_file.xdmf").write(volume_markers)
    f.XDMFFile("boundary_file.xdmf").write(surface_markers)

.. testcode::

    mesh = MeshFromXDMF(volume_file="volume_file.xdmf", boundary_file="boundary_file.xdmf")

.. testoutput::

    Succesfully load mesh with 200 cells

The recommended workflow is to mesh your geometry with your favourite meshing software (`SALOME <https://www.salome-platform.org/?lang=fr>`_, `gmsh <https://gmsh.info/>`_...) and convert the produced mesh with `meshio <https://github.com/nschloe/meshio>`_.

GMSH example
------------

The DOLFINx tutorial gives an `example <https://jorgensd.github.io/dolfinx-tutorial/chapter1/membrane_code.html#creating-the-mesh>`_ of mesh generation with gmsh.

SALOME example
--------------

This is a step-by-step guide to meshing with `SALOME 9.12.0 <https://www.salome-platform.org/>`_.

Building the geometry in SALOME 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Open SALOME and create a new study.
2. Activate the Geometry module

.. thumbnail:: ../images/salome_guide_1.png
    :width: 400
    :align: center

3. Create a first square by clicking "Create rectangular face". Keep the default parameters. Click "Apply and Close"

.. thumbnail:: ../images/salome_guide_2.png
    :width: 400
    :align: center

4. Repeat the operation to create a second square

5. Translate the second square by clicking "Operations/Transformation/Translation"

.. thumbnail:: ../images/salome_guide_3.png
    :width: 400
    :align: center

6. Make sure Face 2 is selected. Enter 100 for the Dx value. Click "Apply and Close"

.. thumbnail:: ../images/salome_guide_4.png
    :width: 400
    :align: center

7. Create a compound by clicking "New Entity/Build/Compound" make sure Face_1 and Translation_1 are selected then click "Apply and Close".

.. thumbnail:: ../images/salome_guide_5.png
    :width: 400
    :align: center

8. Create a group "New Entity/Group/Create group". In Shape Type, select the 2D surface. Name the group "left_volume". Make sure Compound_1 is selected.
Click on the left square and click "Add" (2 should appear in the white window). Click "Apply and Close".

.. thumbnail:: ../images/salome_guide_6.png
    :width: 400
    :align: center

9. Repeat the operation to create a group "right_volume" with the right square (12 should appear in the white window).

10. Create another group "left_boundary" but this time in Shape Type select the 1D curve. Click on the left edge of the left square and click "Add". Click "Apply and Close".

.. thumbnail:: ../images/salome_guide_7.png
    :width: 400
    :align: center

11. Repeat the operation to create a group "right_boundary" with the right edge of the right square. Your study should look like:

.. thumbnail:: ../images/salome_guide_8.png
    :width: 400
    :align: center

12. Click on "Mesh" to activate the mesh module.

.. thumbnail:: ../images/salome_guide_9.png
    :width: 400
    :align: center

13. Create a mesh by clicking "Mesh/Create Mesh".

14. Make sure Compound_1 is selected in "Geometry". Under the 2D tab, select "NETGEN 1D-2D" as algorithm.

.. thumbnail:: ../images/salome_guide_10.png
    :width: 400
    :align: center

15. Next to "Hypothesis" click on the gear symbol. Select "NETGEN 2D Simple Parameters". Click Ok. Click "Apply and Close".

.. thumbnail:: ../images/salome_guide_11.png
    :width: 400
    :align: center

    In the Objet Browser, under Mesh_1 you should see Groups of Edges and Groups of Faces, containing left_boundary, right_boundary, left_volume and right_volume.

16. Export the mesh to MED by right clicking on Mesh_1 in the Object Browser, then Export/MED file. Choose a location where you want to write your MED file and click Save.

.. thumbnail:: ../images/salome_guide_12.png
    :width: 400
    :align: center

17. Convert mesh with meshio (at the time or writing we are using meshio 5.3)

.. code-block:: bash

    python convert_mesh.py

The script `convert_mesh.py` is:

.. code-block:: python

    import meshio

    def convert_med_to_xdmf(
        med_file,
        cell_file="mesh_domains.xdmf",
        facet_file="mesh_boundaries.xdmf",
        cell_type="tetra",
        facet_type="triangle",
    ):
        """Converts a MED mesh to XDMF
        Args:
            med_file (str): the name of the MED file
            cell_file (str, optional): the name of the file containing the
                volume markers. Defaults to "mesh_domains.xdmf".
            facet_file (str, optional): the name of the file containing the
                surface markers.. Defaults to "mesh_boundaries.xdmf".
            cell_type (str, optional): The topology of the cells. Defaults to "tetra".
            facet_type (str, optional): The topology of the facets. Defaults to "triangle".
        Returns:
            dict, dict: the correspondance dict, the cell types
        """
        msh = meshio.read(med_file)

        correspondance_dict = msh.cell_tags

        cell_data_types = msh.cell_data_dict["cell_tags"].keys()

        for mesh_block in msh.cells:
            if mesh_block.type == cell_type:

                meshio.write_points_cells(
                    cell_file,
                    msh.points,
                    [mesh_block],
                    cell_data={"f": [-1 * msh.cell_data_dict["cell_tags"][cell_type]]},
                )
            elif mesh_block.type == facet_type:
                meshio.write_points_cells(
                    facet_file,
                    msh.points,
                    [mesh_block],
                    cell_data={"f": [-1 * msh.cell_data_dict["cell_tags"][facet_type]]},
                )

        return correspondance_dict, cell_data_types


    if __name__ == "__main__":
        filename = "Mesh_1.med"
        correspondance_dict, cell_data_types = convert_med_to_xdmf(
            filename, cell_type="triangle", facet_type="line")
        print(correspondance_dict)

Running this script produces mesh_domains.xdmf, mesh_boundaries.xdmf, mesh_domains.h5, mesh_boundaries.h5 and a dictionary of correspondance between the markers and the mesh entities:

.. code-block:: bash

    {-6: ['left_volume'], -7: ['right_volume'], -8: ['left_boundary'], -9: ['right_boundary']}

The correspondance dictionary can be used to assign the correct markers to the mesh.
Here, the left volume is tagged with ID 6, the right boundary is tagged with ID 9.

18. Inspect the produced XDMF files with Paraview using the XDMF3 S reader. The file mesh_domains.xdmf should look like:

.. thumbnail:: ../images/salome_guide_13.png
    :width: 400
    :align: center


19. Test the mesh in FESTIM by running:

.. code-block:: python

    import festim as F

    model = F.Simulation()

    model.mesh = F.MeshFromXDMF(
        volume_file="mesh_domains.xdmf", boundary_file="mesh_boundaries.xdmf"
    )

    model.materials = [F.Material(D_0=1, E_D=0, id=6), F.Material(D_0=5, E_D=0, id=7)]

    model.boundary_conditions = [
        F.DirichletBC(field="solute", value=1, surfaces=[8]),
        F.DirichletBC(field="solute", value=0, surfaces=[9]),
    ]

    model.T = F.Temperature(823)

    model.exports = [F.XDMFExport("solute")]

    model.settings = F.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        transient=False,
    )

    model.initialise()
    model.run()

20. The simulation should run without errors. The solute field can be visualised with Paraview.

.. thumbnail:: ../images/salome_guide_14.png
    :width: 400
    :align: center

Meshing CAD files in SALOME
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have a CAD model, you can export it to a mesh with SALOME.

1. Create a new study
2. Activate the Geometry module
3. Import STEP file by clicking "File/Import/STEP"

.. thumbnail:: ../images/salome_guide_cad_1.png
    :width: 400
    :align: center

4. By clicking "Fit to selection" you can see the imported geometry:

.. thumbnail:: ../images/salome_guide_cad_2.png
    :width: 400
    :align: center

5. Create a partition just like in the previous example
6. Create groups of volumes and faces
7. Mesh the geometry
8. Export the mesh to MED
9. Convert the mesh to XDMF (don't forget to change the cell and facet types in the script)

------------------
Meshes from FEniCS
------------------

See the `FEniCS documentation <https://fenicsproject.org/olddocs/dolfin/latest/python/demos/built-in-meshes/demo_built-in-meshes.py.html>`_ for more built-in meshes.
