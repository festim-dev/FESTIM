===========
Temperature
===========

.. testsetup::

    from festim import HeatTransferProblem, TemperatureFromXDMF

Definition of a temperature field or problem is essential for hydrogen transport 
and FESTIM as a whole.
Regardless of how you define the temperature of the problem, it is passed to the :code:`T` attribute of the :class:`festim.Simulation` object.

----------------------
Analytical expressions
----------------------

The temperature can be defined as a constant value in Kelvin (K):

.. testcode::

    my_temperature = 300


Temperature can also be defined as an expression of time and/or space.
For example:

.. math::

    T = 300 + 2 x + 3 t 

would be passed to FESTIM as:

.. code-block:: python

    from festim import x, t

    my_temp = 300 + 2*x + 3*t

More complex expressions can be expressed with sympy:

.. math::

    T = \exp(x) \ \sin(t)

would be passed to FESTIM as:

.. testcode::

    from festim import x, t
    import sympy as sp

    my_temp = sp.exp(x) * sp.sin(t)

Conditional expressions are also possible:

.. testcode::

    from festim import x, t
    import sympy as sp

    my_temp = sp.Piecewise((400, t < 10), (300, True))

---------------------------
From a heat transfer solver
---------------------------

Temperature can also be obtained by solving the heat equation.
Users can define heat transfer problems using :class:`festim.HeatTransferProblem`.


.. testcode::

    my_temp = HeatTransferProblem()

For a steady-state problem:

.. code-block:: python

    my_temp = HeatTransferProblem(transient=False)

:ref:`Boundary conditions<boundary conditions>` and :ref:`heat sources<sources>` can then be applied to this heat transfer problem.

For transient problems, an initial condition is required:

.. code-block:: python

    model.T = HeatTransferProblem(
        transient=True,
        initial_condition=300,
    )

Initial conditions can be given as float, sympy expressions or a :class:`festim.InitialCondition` instance in order to read from a XDMF file (see :ref:`Initial Conditions<Initial Conditions>` for more details).

----------------
From a XDMF file
----------------

Temperature can also be read from a XDMF file (see :class:`festim.TemperatureFromXDMF`).

.. code-block:: python

    my_temp = TemperatureFromXDMF('temperature.xdmf', label='temperature')

.. note::

    The XDMF file must contain a scalar field named 'temperature'.
    Moreover, it has to have been exported in "checkpoint" mode (see :ref:`XDMF export`).
