==================
Initial conditions
==================

The initial conditions are essential to transient FESTIM simulations. They describe the mathematical problem at the beginning of the simulation.
By default, the initial conditions are set to zero.
However, it is possible to set the initial conditions with the :class:`festim.InitialCondition` class.

.. testcode::

    import festim as F

    my_ic = F.InitialCondition(value=10, field=0)

The value can also be a function of space:

.. testcode::

    import festim as F
    from festim import x, y, z

    my_ic = F.InitialCondition(value=x**2 + y**2 + z**2, field=0)

Initial conditions can also be read from a previously written XDMF file. This is useful when restarting a simulation.

.. testcode::

    import festim as F

    my_ic = F.InitialCondition(
        value="ic_file.xdmf",
        label="mobile",
        time_step=-1,
        field=0
    )

In the snippet above, the initial condition is read from the file ``ic_file.xdmf``.
The label ``mobile`` is used to identify the mesh in the file.
The timestep ``-1`` indicates that the last timestep of the file should be read.

.. note::

    The XDMF file must be readable. To do so, the XDMF file must be created with checkpointing. See :class:`festim.XDMFExport`.
    For more information on checkpointing in FEniCS, see `this page <https://fenicsproject.discourse.group/t/loading-xdmf-data-back-in/1925/4>`_.
