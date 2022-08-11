===================
Boundary conditions
===================

The boundary conditions (BCs) are essential to FESTIM simulations. They describe the mathematical problem at the boundaries of the simulated domain.
If no BC is set on a boundary, it is assumed that the flux is null. This is also called a symmetry BC.

---------------
Basic BCs
---------------
These BCs can be used for heat transfer or hydrogen transport simulations.

Imposing the solution
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    my_bc = DirichletBC(surfaces=[2, 4], value=10, field=0)

.. admonition:: Note
   :class: tip

    Here, we set `field` to `0` to specify this BC applies to the mobile hydrogen concentration. `1` would stand for the trap 1 concentration and `"T"` for temperature.

The `value` argument can be space and time dependent by making use of the FESTIM variables ``x``, ``y``, ``z`` and ``t``:

.. code-block:: python

    from festim import x, y, z, t
    my_bc = DirichletBC(surfaces=3, value=10 + x**2 + t, field="T")


To use more complicated mathematical expressions, you can use the sympy package:

.. code-block:: python

    from festim import x, y, z, t
    import sympy as sp

    my_bc = DirichletBC(surfaces=3, value=10*sp.exp(-t), field="T")

- CustomDirichlet

The value of the concentration field can be temperature-dependent (useful when dealing with heat-transfer solvers) with `CustomDirichlet`:

.. code-block:: python

    def value(T):
        return 3*T + 2

    my_bc = CustomDirichlet(surfaces=3, function=value, field=0)

Imposing the flux
^^^^^^^^^^^^^^^^^

When the flux needs to be imposed on a boundary, use the `FluxBC` class.


.. code-block:: python

    my_bc = FluxBC(surfaces=3, value=10 + x**2 + t, field="T")


As for the Dirichlet boundary conditions, the flux can be dependent on temperature and mobile hydrogen concentration:

.. code-block:: python

    def value(T, mobile):
        return mobile**2 + T

    my_bc = CustomFlux(surfaces=3, function=value, field=0)


---------------
Hydrogen transport BCs
---------------

- RecombinationFlux
- SievertsLaw
- HenrysLaw
- ImplantationDirichlet (refer to Theory section)

-----------------
Heat transfer BCs
-----------------

- ConvectiveFlux
