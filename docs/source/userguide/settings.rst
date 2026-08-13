.. _settings_ug:

========
Settings
========

The settings of a FESTIM simulation are defined with a :class:`festim.Settings` object.

.. testcode::

    import festim as F

    my_settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
        transient=False,
    )


Here you define with:

* ``transient``: whether the simulation is transient or steady-state
* ``final_time``: the final time of the simulation
* ``chemical_pot``: whether to run the simulation with conservation of chemical potential at interfaces (only useful for multi-materials)
* ``absolute_tolerance``: the absolute tolerance of the Newton solver
* ``relative_tolerance``: the relative tolerance of the Newton solver
* ``maximum_iterations``: the maximum iterations of the Newton solver

.. note::
    The Soret effect is not a setting. It is a drift term passed to the problem's
    ``drift_terms``, alongside advection and electromigration -- see
    :class:`festim.SoretTerm` and the :doc:`theory guide </theory>`.

