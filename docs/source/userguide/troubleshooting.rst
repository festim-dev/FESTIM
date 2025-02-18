===============
Troubleshooting
===============

----------------------
Where can I find help?
----------------------

If you're having an issue, the best way to find help is to join the `FESTIM discourse <https://festim.discourse.group>`_. Engage in discussions, ask questions, and connect with other users. This collaborative space allows you to share your experiences and seek assistance.

-------------
Common issues
-------------

Although FESTIM is designed to be easy to use, users may encounter some common issues. This section provides some guidance on how to resolve these issues.
We are simply solving a set of equations using the finite element method, and are, therefore, subject to the same issues that other FEM codes face.

^^^^^^^^^^^^^^^^^^^^^^^
Solver doesn't converge
^^^^^^^^^^^^^^^^^^^^^^^

The first thing to check is the details of the Newton solver iterations.
To do so, you must set the ``log_level`` to ``20`` (default is ``40``).
This will provide more information during the solving stage.

.. testcode::

    import dolfinx

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

From there, depending on the behaviour of the solver, you can try the following:

- if the solver diverges, try reducing the time step and/or mesh refinement. It is often helpful to inspect the fields to see if there are any obvious issues (like lack of refinement).
- If the solver converges to a value above the tolerance, try increasing the tolerance. Sometimes, the absolute tolerance is too low for the problem at hand, especially when dealing with large numbers.


^^^^^^^^^^^^^^^^^^^^^^^^^^^
Solution is zero everywhere
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes, the solver converges fine but the solution is zero everywhere.
This is often due to an excessively high absolute tolerance.
The Newton solver then converges in zero iterations. In other words, nothing is solved.
First, check that this is the case by setting the log level to INFO:

.. testcode::

    import dolfinx

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

Then increase the absolute tolerance of the solver:

.. testcode::

    import festim as F

    my_model = F.HydrogenTransportProblem()
    my_model.settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
    )