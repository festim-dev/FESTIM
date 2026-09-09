.. _initial_conditions_guide:

==================
Initial conditions
==================

.. testsetup:: ICs

    import festim as F

    my_mat = F.Material(D_0=1, E_D=0.1)
    my_vol = F.VolumeSubdomain(id=1, material=my_mat)
    H = F.Species(name="H")

The initial conditions are essential to transient FESTIM simulations. They describe the mathematical problem at the beginning of the simulation.
By default, the initial conditions are set to zero.

All initial conditions in FESTIM require a volume subdomain, defined with the :class:`festim.VolumeSubdomain` class (see :ref:`Volume Subdomains`).
They are passed to the problem as a list in its ``initial_conditions`` attribute.

.. testcode:: ICs

    my_model = F.HydrogenTransportProblem()

    my_model.initial_conditions = [
        F.InitialConcentration(value=1e20, species=H, volume=my_vol),
    ]

----------------------
Initial concentration
----------------------

The initial concentration of a species is set with the :class:`festim.InitialConcentration` class.

.. testcode:: ICs

    from festim import InitialConcentration

    my_ic = InitialConcentration(value=1e20, species=H, volume=my_vol)

The value can also be a function of space and temperature:

.. testcode:: ICs

    from festim import InitialConcentration

    my_custom_value = lambda x, T: 1e20 * x[0] ** 2 + 1e15 * T

    my_ic = InitialConcentration(value=my_custom_value, species=H, volume=my_vol)

.. note::

    When defining custom functions for values, only the arguments :code:`x` and :code:`T` can be defined.
    Spatial coordinates can be referred to by their indices, such as :code:`x[0]`, :code:`x[1]`, and :code:`x[2]`, regardless of the coordinate system used.
    An initial condition cannot be a function of time: passing a function of :code:`t` raises an error.

In a multi-material problem, one :class:`festim.InitialConcentration` is needed per volume subdomain where the initial concentration is not zero.

--------------------
Initial temperature
--------------------

For transient :class:`festim.HeatTransferProblem` simulations, the initial temperature is set with the :class:`festim.InitialTemperature` class and passed to the ``initial_condition`` attribute of the heat transfer problem (see :ref:`Temperature <temperature_guide>`).

.. testcode:: ICs

    from festim import InitialTemperature

    my_heat_model = F.HeatTransferProblem()

    my_heat_model.initial_condition = InitialTemperature(value=300, volume=my_vol)

The value can be a function of space:

.. testcode:: ICs

    from festim import InitialTemperature

    my_ic = InitialTemperature(value=lambda x: 300 + 100 * x[0], volume=my_vol)

----------------------------
Restarting from a checkpoint
----------------------------

Initial conditions can also be read from a previously written checkpoint file. This is useful when restarting a simulation.
The field is read with :func:`festim.read_function_from_file` and passed as the value of the initial condition:

.. code-block:: python

    import festim as F

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1e-3, 100))

    my_ic = F.InitialConcentration(
        value=F.read_function_from_file(
            filename="state.bp", name="H", timestamp=10.0, mesh=my_model.mesh.mesh
        ),
        species=H,
        volume=my_vol,
    )

In the snippet above, the initial condition is read from the file ``state.bp``.
The name ``H`` is the name under which the field was written (the name of the species) and the timestamp ``10.0`` is the time at which it was written.
The ``mesh`` argument is the mesh of the new simulation: the field is interpolated onto it, which also allows restarting on a different mesh than the one the checkpoint was written on.

.. note::

    Only files written with an export in ``"checkpoint"`` format can be read back this way (see :ref:`Checkpoints`).
    The visualisation formats (``"vtx"``, ``"vtkhdf"``, ``"xdmf"``) store values interpolated onto the mesh nodes and cannot be used to restart a simulation.

If the checkpoint was written with ``backend="h5py"``, the same ``backend`` must be passed to :func:`festim.read_function_from_file`.
