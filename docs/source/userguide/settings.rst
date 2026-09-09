.. _settings_ug:

========
Settings
========

.. testsetup:: settings

    import festim as F

The settings of a FESTIM simulation are defined with a :class:`festim.Settings` object.

.. testcode:: settings

    import festim as F

    my_settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
        transient=False,
    )

Here you define with:

* ``atol``: the absolute tolerance of the Newton solver
* ``rtol``: the relative tolerance of the Newton solver
* ``max_iterations``: the maximum number of iterations of the Newton solver (default: 30)
* ``transient``: whether the simulation is transient or steady-state (default: ``True``)
* ``final_time``: the final time of a transient simulation
* ``stepsize``: the stepsize of a transient simulation, as a :class:`festim.Stepsize` or a number (see :ref:`Stepsize <stepsize_guide>`)
* ``element_degree``: the degree of the finite elements used for the concentrations (default: 1)

For a transient simulation, ``final_time`` and ``stepsize`` are required:

.. testcode:: settings

    my_settings = F.Settings(
        atol=1e10,
        rtol=1e-10,
        final_time=100,
        stepsize=F.Stepsize(initial_value=1),
    )

The settings are then passed to the ``settings`` attribute of the problem:

.. testcode:: settings

    my_model = F.HydrogenTransportProblem()
    my_model.settings = my_settings

See :class:`festim.Settings` for more details.

----------
Tolerances
----------

The Newton solver stops when the residual of the equations is below ``atol`` or when it has decreased by a factor ``rtol`` compared to its initial value.
Since concentrations in FESTIM are expressed in :math:`\mathrm{m}^{-3}`, the residual can be very large in absolute terms, and ``atol`` should be chosen accordingly (see :ref:`Troubleshooting`).

-----------------------
Advanced: PETSc options
-----------------------

The non-linear solver is a `PETSc SNES <https://petsc.org/release/manual/snes/>`_ solver.
By default, FESTIM uses a Newton method without line search and a direct LU solver (MUMPS when available) for the linear systems.
These defaults can be overridden by passing a dictionary of `PETSc options <https://petsc.org/release/manual/snes/#general-options>`_ to the ``petsc_options`` argument of the problem:

.. testcode:: settings

    my_model = F.HydrogenTransportProblem(
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "ksp_type": "gmres",
            "pc_type": "hypre",
        }
    )

.. note::

    The options ``snes_atol``, ``snes_rtol`` and ``snes_max_it`` are always taken from the :class:`festim.Settings` object (``atol``, ``rtol`` and ``max_iterations``) and cannot be set through ``petsc_options``.

------------
Progress bar
------------

A progress bar is displayed during transient simulations. It can be turned off with:

.. testcode:: settings

    my_model.show_progress_bar = False
