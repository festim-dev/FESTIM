===============
Post-processing
===============

.. warning::

    🔨 This page is under construction. 🔨

Exporting fields
----------------

Field exports write a whole field -- a species concentration, the temperature, a custom
expression -- to a file at each timestep. The class says *what* to export and the
``format`` argument says *how* to write it:

.. code-block:: python

    import festim as F

    my_model.exports = [
        F.SpeciesExport("results.bp", field=[H], subdomain=my_volume),
        F.TemperatureExport("temperature.bp"),
    ]

Available formats
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 12 20 53

   * - ``format``
     - Extension
     - Opens in ParaView
     - Notes
   * - ``"vtx"``
     - ``.bp``
     - yes
     - The default. Written by DOLFINx as an ADIOS2 directory of files.
   * - ``"vtkhdf"``
     - ``.vtkhdf``
     - yes
     - A single HDF5 file rather than a directory. Several exports may share one
       filename, becoming separate blocks of the same file.
   * - ``"xdmf"``
     - ``.xdmf``
     - yes
     - Writes an ``.xdmf`` file next to an ``.h5`` file holding the data.
   * - ``"checkpoint"``
     - ``.bp`` / ``.h5``
     - no
     - For restarting a simulation, not for viewing. See below.

The visualisation formats interpolate the field onto the mesh nodes. ``"checkpoint"``
stores it in its own function space instead, so it can be read back exactly.

One file for a multi-material model
-----------------------------------

In a :class:`festim.HydrogenTransportProblemDiscontinuous`, each subdomain has its own
mesh, so each export normally produces its own file. The ``"vtkhdf"`` format lets them
share one, with each subdomain stored as a named block:

.. code-block:: python

    my_model.exports = [
        F.SpeciesExport("results.vtkhdf", field=[H], subdomain=vol1, format="vtkhdf"),
        F.SpeciesExport("results.vtkhdf", field=[H], subdomain=vol2, format="vtkhdf"),
    ]

.. note::

    Writing ``.vtkhdf`` files relies on ``h5py``. To write them from more than one MPI
    process, ``h5py`` must be built with MPI support::

        conda install -c conda-forge "h5py=*=mpi_*"

    FESTIM raises an error explaining this if it is missing.

Checkpoints
-----------

A checkpoint stores the field in its own function space, so it can be read back exactly
-- and on a different number of processes than it was written on:

.. code-block:: python

    my_model.exports = [
        F.SpeciesExport("state.bp", field=[H], subdomain=vol, format="checkpoint"),
    ]

Read it back into another simulation with
:func:`festim.read_function_from_file`, as an initial condition:

.. code-block:: python

    my_other_model.initial_conditions = [
        F.InitialConcentration(
            value=F.read_function_from_file(
                filename="state.bp", name="H", timestamp=10.0
            ),
            species=H,
            volume=vol,
        )
    ]

Pass ``backend="h5py"`` to write a plain ``.h5`` file instead of the ADIOS2 default.
The same ``backend`` must then be given when reading it back::

    F.read_function_from_file(
        filename="state.h5", name="H", timestamp=10.0, backend="h5py"
    )

Only checkpoints can be read back this way. The visualisation formats store values
interpolated onto the mesh nodes rather than the degrees of freedom, so
:func:`festim.read_function_from_file` cannot restart a simulation from them.

Exporting at chosen times
-------------------------

By default a field is written at every timestep. Pass ``times`` to write only at
specific ones; those times are added to the stepsize milestones so they are hit exactly:

.. code-block:: python

    F.SpeciesExport("results.bp", field=[H], subdomain=vol, times=[1, 10, 100])

Deprecated export classes
-------------------------

Before formats were selectable, each format had its own class. These still work but
emit a :class:`DeprecationWarning` and will be removed in a future release:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Deprecated
     - Use instead
   * - ``F.VTXSpeciesExport(...)``
     - ``F.SpeciesExport(..., format="vtx")``
   * - ``F.VTXTemperatureExport(...)``
     - ``F.TemperatureExport(..., format="vtx")``
   * - ``F.XDMFExport(...)``
     - ``F.SpeciesExport(..., format="xdmf")``
   * - ``F.CustomFieldExport(..., checkpoint=True)``
     - ``F.CustomFieldExport(..., format="checkpoint")``
