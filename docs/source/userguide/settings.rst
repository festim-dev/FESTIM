========
Settings
========

The settings of a FESTIM simulation are defined with a :code:`Settings` object.

.. code-block:: python

    import festim as F

    my_settings = F.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-10,
        transient=False,
        chemical_pot=True,
    )


Here you define:

* wether the simulation is transient or steady-state
* the final time of the simulation
* wether to run the simulation with conservation of chemical potential at interfaces (only useful for multi-materials)
* wether to turn the Soret effect on
* the absolute and relative tolerance of the Newton solver
* the maximum iterations of the Newton solver

More advanced settings are also available:

* the type of finite elements for traps (DG elements can be useful to account for discontinuities)
* Wether to update the jacobian at each iteration or not
* the linear solver
