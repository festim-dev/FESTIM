.. _stepsize_guide:

========
Stepsize
========

.. testsetup:: stepsize

    import festim as F

For transient problems, a :class:`festim.Stepsize` is required.
It represents the time discretisation of the problem and is passed to the ``stepsize`` argument of :class:`festim.Settings` (see :ref:`Settings <settings_ug>`).
Here is an example creating a stepsize of 1.2 seconds:

.. testcode:: stepsize

    import festim as F

    my_stepsize = F.Stepsize(initial_value=1.2)

    my_settings = F.Settings(atol=1e10, rtol=1e-10, final_time=100, stepsize=my_stepsize)

.. note::

    If your stepsize is constant, you can define it simply as a ``float`` or ``int``:

    .. testcode:: stepsize

        my_settings = F.Settings(atol=1e10, rtol=1e-10, final_time=100, stepsize=1.2)

----------------------
Adaptive time stepping
----------------------

To use the adaptive time stepping implemented in FESTIM, the arguments ``growth_factor``, ``cutback_factor`` and ``target_nb_iterations`` need to be set.

.. testcode:: stepsize

    my_stepsize = F.Stepsize(
        initial_value=1.2,
        growth_factor=1.2,
        cutback_factor=0.8,
        target_nb_iterations=4,
    )

When doing so, the stepsize will be multiplied by ``growth_factor`` when the Newton solver converges in fewer than ``target_nb_iterations`` iterations, and multiplied by ``cutback_factor`` when it needs more than ``target_nb_iterations`` iterations to converge.
``growth_factor`` must be greater than one and ``cutback_factor`` smaller than one.
This is extremely useful when solving transient problems with a large time range, as the time step will be large when the solution is smooth and small when the solution is changing rapidly.

.. note::

    Unlike FESTIM 1.x, the stepsize is not automatically reduced when the solver fails to converge: the simulation stops with an error instead.
    See :ref:`Troubleshooting` for how to inspect the Newton iterations.

Another option for controlling the stepsize is to use the ``max_stepsize`` parameter. This parameter defines the maximal value of the stepsize during simulations,
and it can be set as a constant or a callable function of time:

.. testcode:: stepsize

    def max_stepsize(t):
        if t <= 5:
            return 1.5
        elif t > 5 and t < 10:
            return 2.5
        else:
            return None

    my_stepsize = F.Stepsize(
        initial_value=1.2,
        growth_factor=1.2,
        cutback_factor=0.8,
        target_nb_iterations=4,
        max_stepsize=max_stepsize,
    )

Returning ``None`` means the stepsize is not capped at this time.

----------
Milestones
----------

The ``milestones`` argument can be used to make sure the simulation passes through specific times.
This will modify the stepsize as needed.

.. testcode:: stepsize

    my_stepsize = F.Stepsize(
        initial_value=1.2,
        growth_factor=1.2,
        cutback_factor=0.8,
        target_nb_iterations=4,
        max_stepsize=5,
        milestones=[1, 5, 6, 10],
    )

Milestones are only relevant for adaptive stepsizes: an error is raised if they are given without ``growth_factor`` and ``cutback_factor``.
The times given to the ``times`` argument of the exports (see :ref:`Exporting at chosen times`) are automatically added to the milestones.

The ``milestone_tolerance`` argument (default ``1e-5``) is the relative tolerance used to decide whether the current time is close enough to a milestone to count as having reached it.
