.. _settings_ug:

========
Settings
========

The settings of a FESTIM simulation are defined with a :class:`festim.Settings` object.

.. testcode::

    import festim as F

    my_settings = F.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-10,
        transient=False,
        chemical_pot=True,
    )


Here you define with:

* ``transient``: wether the simulation is transient or steady-state
* ``final_time``: the final time of the simulation
* ``chemical_pot``: wether to run the simulation with conservation of chemical potential at interfaces (only useful for multi-materials)
* ``soret``: wether to turn the Soret effect on or not
* ``absolute_tolerance``: the absolute tolerance of the Newton solver
* ``relative_tolerance``: the relative tolerance of the Newton solver
* ``maximum_iterations``: the maximum iterations of the Newton solver

More advanced settings are also available:

* ``traps_element_type``: the type of finite elements for traps (DG elements can be useful to account for discontinuities)
* ``update_jacobian``: wether to update the jacobian at each iteration or not
* ``linear_solver``: linear solver method for the Newton solver
* ``preconditioner``: preconditioning method for the Newton solver

See :ref:`settings_api` for more details.