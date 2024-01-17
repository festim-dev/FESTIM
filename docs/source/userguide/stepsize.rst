Stepsize
========


For transient problems, a :class:`festim.Stepsize` is required.
It represents the time discretisation of the problem.
Here is an example to create a stepsize of 1.2 seconds:

.. code-block:: python

    import festim as F
    my_stepsize = F.Stepsize(initial_value=1.2)

To make use of the adaptive time stepping implemented in FESTIM, the arguments ``stepsize_change_ratio`` needs to be set to a value above 1.

.. code-block:: python

    my_stepsize = F.Stepsize(initial_value=1.2, stepsize_change_ratio=1.5)

When doing so, the stepsize will grow according to this ratio when the numerical solver converges in 4 iterations or less.
It will shrink by the same ratio when the solver needs more than 4 iterations to converge.
This is extremely useful when solving transient problems with a large time range, as the time step will be large when the solution is smooth and small when the solution is changing rapidly.
Moreover, if the solver doesn't converge, the stepsize will be reduced and the solver will be called again.
Setting the ``dt_min`` argument will prevent the stepsize to become too small and will stop the simulation when this happens.
To cap the stepsize after some time, the parameters ``t_stop`` and ``stepsize_stop_max`` can be used.

.. code-block:: python

    my_stepsize = F.Stepsize(initial_value=1.2, stepsize_change_ratio=1.5, dt_min=1e-6, t_stop=10, stepsize_stop_max=1.5)

To make sure the simulation passes through specific times, the ``milestones`` argument can be used.
This will modify the stepsize as needed.

.. code-block:: python

    my_stepsize = F.Stepsize(
        initial_value=1.2,
        stepsize_change_ratio=1.5,
        dt_min=1e-6,
        t_stop=10,
        stepsize_stop_max=1.5,
        milestones=[1, 5, 6, 10]
        )