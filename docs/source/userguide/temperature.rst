===========
Temperature
===========

Definition of a temperature field or problem is essential for hydrogen transport 
and FESTIM as a whole. 

----------------------
Analytical expressions
----------------------

The temperature can be defined as a constant value in Kelvin (K):

.. code-block:: python

    my_temperature = Temperature(value=300)


Temperature can also be defined as an expression of time and/or space.
For example:

.. math::

    T = 300 + 2 x + 3 t 

would be passed to FESTIM as:

.. code-block:: python

    from festim import x, t

    my_temp = Temperature(300+2*x+3*t)

More complex expressions can be expressed with sympy:

.. math::

    T = \exp(x) \ \sin(t)

would be passed to FESTIM as:

.. code-block:: python

    from festim import x, t
    import sympy as sp

    my_temp = Temperature(sp.exp(x)*sp.sin(t))

For more details, see :class:`festim.Temperature`.

---------------------------
From a heat transfer solver
---------------------------

Temperature can also be obtained by solving the heat equation.
Users can define heat transfer problems using :class:`festim.HeatTransferProblem`.


.. code-block:: python

    my_temp = HeatTransferProblem()

For a steady-state problem:

.. code-block:: python

    my_temp = HeatTransferProblem(transient=False)

:ref:`Boundary conditions<boundary conditions>` and :ref:`heat sources<sources>` can then be applied to this heat transfer problem.

----------------
From a XDMF file
----------------

Temperature can also be read from a XDMF file (see :class:`festim.TemperatureFromXDMF`).

.. code-block:: python

    my_temp = TemperatureFromXDMF('temperature.xdmf', label='temperature')

.. note::

    The XDMF file must contain a scalar field named 'temperature'.
    Moreover, it has to have been exported in "checkpoint" mode (see :ref:`XDMF export`).
