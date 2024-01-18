==================
Initial conditions
==================

The initial conditions are essential to transient FESTIM simulations. They describe the mathematical problem at the beginning of the simulation.
By default, the initial conditions are set to zero.
However, it is possible to set the initial conditions with the :class:`festim.InitialCondition` class.

.. code-block:: python

    import festim as F

    my_ic = F.InitialCondition(value=10, field=0)

The value can also be a function of space:

.. code-block:: python

    import festim as F
    from festim import x, y, z

    my_ic = F.InitialCondition(value=x**2 + y**2 + z**2, field=0)