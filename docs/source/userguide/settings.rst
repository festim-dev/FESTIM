========
Settings
========

The settings of a FESTIM simulation are defined with a :code:`Settings` object.

.. code-block:: python

    import festim as F

    my_settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
        transient=False,
    )


Here you define:

* wether the simulation is transient or steady-state
* the final time of the simulation
* wether to run the simulation with conservation of chemical potential at interfaces (only useful for multi-materials)
* wether to turn the Soret effect on
* the absolute and relative tolerance of the Newton solver
* the maximum iterations of the Newton solver

