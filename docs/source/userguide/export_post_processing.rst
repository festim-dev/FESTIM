===============
Post-processing
===============

Exports are added to the simulation object as a list of :class:`festim.Export` objects:

.. code-block:: python

    import festim as F

    my_model = F.Simulation()

    my_model.exports = [..., ...]

-------------------
Exporting solutions
-------------------

^^^^^^^^^^^
XDMF export
^^^^^^^^^^^

The most straightforward way to export solutions (concentrations, temperature) with FESTIM is to use the :class:`festim.XDMFExport` class.
This class leverages the ``XDMFFile`` class of ``fenics`` and allows solutions to be exported in the XDMF format.
The following example shows how to export the solution of a 1D problem:

.. code-block:: python

    import festim as F

    my_model = F.Simulation()
    my_model.mesh = F.MeshFromVertices([0, 1, 2, 3])
    my_model.materials = F.Material(id=1, D_0=1, E_D=0)
    my_model.boundary_conditions = [
        F.DirichletBC(surfaces=[1, 2], value=0, field="solute"),
    ]
    my_model.sources = [F.Source(value=1, volume=1, field="solute")]

    my_model.T = F.Temperature(500)

    my_model.exports = [
        F.XDMFExport(
            "solute",  # the field we want to export
            label="mobile",  # how the field will be labelled in the XDMF file
            filename="./mobile_conc.xdmf",
            checkpoint=False,  # needed in 1D
        )
    ]

    my_model.settings = F.Settings(
        absolute_tolerance=1e-10, relative_tolerance=1e-10, transient=False
    )

    my_model.initialise()
    my_model.run()


Running this should produce a file called ``mobile_conc.xdmf`` in the current directory.
The file can then be opened in Paraview or any other software that can read XDMF files.

For transient simulations, by default, :class:`festim.XDMFExport` will export the solution at each timestep.
It is possible to change this behaviour to limit the number of times the file is written to.
By setting the ``mode`` attribute to ``10``, for example, the solution will be exported every 10 timesteps.
Setting it to ``last`` will export the solution only at the last timestep.

.. code-block:: python

    my_model.exports = [
        F.XDMFExport(
            "solute",
            label="mobile",
            filename="./mobile_conc.xdmf",
            checkpoint=False,
            mode=10,
        )
    ]

The ``checkpoint`` attribute must be set to ``True`` for the XDMF file to be readable by Paraveiw.

^^^^^^^^^^^^^^^
TXT export (1D)
^^^^^^^^^^^^^^^

The ``TXTExport`` class allows solutions to be exported in a simple text format.
It works in 1D only. For multi-dimensional problems, use the :class:`festim.XDMFExport` class instead.

.. code-block:: python

    import festim as F

    my_export = F.TXTExport(field="solute", filename="./mobile_conc.txt")

Adding this export to the simulation object will produce a file called ``mobile_conc.txt`` in the current directory.
This file will contain the solution of the ``solute`` field at the degrees of freedom of the mesh and at each timestep.

To only export at specific times in the simulation, use the ``times`` argument:

.. code-block:: python

    my_export = F.TXTExport(
        field="solute", filename="./mobile_conc.txt", times=[0, 1, 2, 3]
    )

^^^^^^^^^^^
Point value
^^^^^^^^^^^

If information about the solution at a specific point is needed, the :class:`festim.PointValue` class can be used.
It is implemented as a derived quantity. See :ref:`Derived quantities` for more information. Here are a few examples:

.. code-block:: python

    import festim as F

    my_export = F.PointValue(field="solute", x=[0.5, 0.5, 0.5])
    my_export = F.PointValue(field="solute", x=(0.5, 0.5, 0.5))
    my_export = F.PointValue(field="solute", x=[0.5, 0.5])
    my_export = F.PointValue(field="solute", x=[0.5])
    my_export = F.PointValue(field="solute", x=0.5)

------------------
Derived quantities
------------------

In addition to exporting the actual solutions, it is possible to export derived quantities.
For instance, you may want to compute the flux of mobile particles at a given boundary.

First, you want to create a :class:`festim.DerivedQuantities` object. This will encompass all the derived quantities you want to compute.
Then, you can add the derived quantities you want to compute to this object.
Finally, you can add the :class:`festim.DerivedQuantities` object to the simulation object.

.. code-block:: python

    my_derived_quantities = F.DerivedQuantities(
        [
            F.SurfaceFlux(field="solute", surface=3),
            F.SurfaceFlux(field="T", surface=1),
            F.AverageVolume(field="retention", volume=1),
            F.TotalVolume(field="retention", volume=2),
        ]
    )

    my_model.exports = [my_derived_quantities, ....]


The complete list of derived quantities can be found at: :ref:`Exports`.

The data can be accessed in three different ways:
- directly from the :class:`festim.DerivedQuantities` (plural) object:

.. code-block:: python

    my_derived_quantities = F.DerivedQuantities(
        [
            F.SurfaceFlux(field="solute", surface=3),
            F.SurfaceFlux(field="T", surface=1),
            F.AverageVolume(field="retention", volume=1),
            F.TotalVolume(field="retention", volume=2),
        ]
    )

    my_model.exports = [my_derived_quantities, ....]

    my_model.initialise()
    my_model.run()

    print(my_derived_quantities.t)
    print(my_derived_quantities.data)

- from the :class:`festim.DerivedQuantity` (singular) object (eg. ``F.SurfaceFlux(...)``):

.. code-block:: python

    flux_surf_3 = F.SurfaceFlux(field="solute", surface=3)

    my_derived_quantities = F.DerivedQuantities(
        [
            flux_surf_3,
            F.SurfaceFlux(field="T", surface=1),
            F.AverageVolume(field="retention", volume=1),
            F.TotalVolume(field="retention", volume=2),
        ]
    )

    my_model.exports = [my_derived_quantities, ....]

    my_model.initialise()
    my_model.run()

    print(flux_surf_3.t)
    print(flux_surf_3.data)
    print(my_derived_quantities.derived_quantities[2].data)

- export and read from a .csv file:

.. code-block:: python

    my_derived_quantities = F.DerivedQuantities(
        [
            F.SurfaceFlux(field="solute", surface=3),
            F.SurfaceFlux(field="T", surface=1),
            F.AverageVolume(field="retention", volume=1),
            F.TotalVolume(field="retention", volume=2),
        ],
        filename="./my_derived_quantities.csv",
    )

    my_model.exports = [my_derived_quantities, ....]

    my_model.initialise()
    my_model.run()


By default, the derived quantities will be computed at each timestep and exported at the last timestep.
This behaviour can be changed by setting the ``nb_iterations_between_compute`` and ``nb_iterations_between_exports`` attributes of the :class:`festim.DerivedQuantities` object.

.. code-block:: python

    my_derived_quantities = F.DerivedQuantities(
        [
            F.SurfaceFlux(field="solute", surface=3),
            F.SurfaceFlux(field="T", surface=1),
            F.AverageVolume(field="retention", volume=1),
            F.TotalVolume(field="retention", volume=2),
        ],
        filename="./my_derived_quantities.csv",
        nb_iterations_between_compute=3,  # compute quantities every 3 timesteps
        nb_iterations_between_exports=10,  # export every 10 timesteps
    )
