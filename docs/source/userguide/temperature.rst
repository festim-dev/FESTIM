.. _temperature_guide:

===========
Temperature
===========

.. testsetup:: temperature

    import festim as F

    my_mat = F.Material(D_0=1, E_D=0.1)
    my_vol = F.VolumeSubdomain(id=1, material=my_mat)
    my_model = F.HydrogenTransportProblem()

Definition of a temperature field or problem is essential for hydrogen transport and FESTIM as a whole, since diffusivities, solubilities and reaction rates all depend on it.
Regardless of how you define the temperature of the problem, it is passed (in Kelvin) to the :code:`temperature` attribute of the :class:`festim.HydrogenTransportProblem` object.

----------------------
Analytical expressions
----------------------

The temperature can be defined as a constant value in Kelvin (K):

.. testcode:: temperature

    my_model.temperature = 300

Temperature can also be defined as a function of time and/or space.
For example:

.. math::

    T = 300 + 2 x + 3 t

would be passed to FESTIM as:

.. testcode:: temperature

    my_model.temperature = lambda x, t: 300 + 2 * x[0] + 3 * t

.. note::

    Only the arguments :code:`x` and :code:`t` can be used.
    Spatial coordinates can be referred to by their indices, such as :code:`x[0]`, :code:`x[1]`, and :code:`x[2]`, regardless of the coordinate system used.

More complex expressions are written with ``ufl`` functions.
For instance:

.. math::

    T = 300 + 100 \exp(-x) \ \sin(t)

would be passed to FESTIM as:

.. testcode:: temperature

    import ufl

    my_model.temperature = lambda x, t: 300 + 100 * ufl.exp(-x[0]) * ufl.sin(t)

Conditional expressions are also possible.
For a function of time only, a plain Python conditional works:

.. testcode:: temperature

    my_model.temperature = lambda t: 400 if t < 10 else 300

For a function of space, use :code:`ufl.conditional`:

.. testcode:: temperature

    import ufl

    my_model.temperature = lambda x: ufl.conditional(ufl.lt(x[0], 0.5), 400, 300)

.. note::

    A function of :code:`t` only is evaluated once per time step and must return a number, so plain Python conditionals can be used.
    A function of :code:`x` is evaluated symbolically on the mesh and must use ``ufl`` operators (:code:`ufl.conditional`, :code:`ufl.exp`...) rather than Python conditionals or ``numpy`` functions.

---------------------------
From a heat transfer solver
---------------------------

Temperature can also be obtained by solving the heat equation (see the :doc:`theory guide </theory>`).
Users can define heat transfer problems using :class:`festim.HeatTransferProblem`.
It takes the same kind of arguments as :class:`festim.HydrogenTransportProblem`: a mesh, subdomains, boundary conditions, sources, exports and settings.
The materials of the volume subdomains must define ``thermal_conductivity``, and for transient problems ``density`` and ``heat_capacity`` too.

.. testcode:: temperature

    import festim as F

    tungsten = F.Material(
        D_0=4.1e-7,
        E_D=0.39,
        thermal_conductivity=170,
        density=19300,
        heat_capacity=130,
    )
    my_volume = F.VolumeSubdomain1D(id=1, borders=[0, 1e-3], material=tungsten)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1e-3)

    my_heat_model = F.HeatTransferProblem(
        mesh=F.Mesh1D(vertices=[0, 2.5e-4, 5e-4, 7.5e-4, 1e-3]),
        subdomains=[my_volume, left, right],
        boundary_conditions=[
            F.HeatFluxBC(subdomain=left, value=1e7),
            F.FixedTemperatureBC(subdomain=right, value=373),
        ],
        settings=F.Settings(atol=1e-8, rtol=1e-10, transient=False),
    )

Heat transfer :ref:`boundary conditions<boundary_conditions>` (:class:`festim.FixedTemperatureBC`, :class:`festim.HeatFluxBC`) and :ref:`heat sources<sources_guide>` (:class:`festim.HeatSource`) can be applied to the heat transfer problem.

For transient problems, an initial condition is required (see :ref:`Initial conditions <initial_conditions_guide>`):

.. testcode:: temperature

    my_heat_model.initial_condition = F.InitialTemperature(value=300, volume=my_volume)
    my_heat_model.settings = F.Settings(
        atol=1e-8, rtol=1e-10, final_time=10, stepsize=0.1
    )

Steady-state temperature
-------------------------

If the temperature field does not evolve in time, the heat transfer problem is solved first and its solution is passed to the hydrogen transport problem as a :code:`dolfinx.fem.Function`:

.. code-block:: python

    my_heat_model.initialise()
    my_heat_model.run()

    my_model.temperature = my_heat_model.u

.. note::

    For this to work, both problems must be defined on the same mesh object (pass the same :class:`festim.Mesh` instance to both).
    Problems defined on different meshes can only be coupled with :class:`festim.CoupledTransientHeatTransferHydrogenTransport` (see below).

Coupled transient problems
--------------------------

When the temperature and the hydrogen concentrations must evolve together, the two problems are coupled with :class:`festim.CoupledTransientHeatTransferHydrogenTransport`.
Both problems must be transient and have the same ``final_time``; the coupled problem takes care of stepping them together and of passing the temperature to the hydrogen transport problem.

.. code-block:: python

    coupled_model = F.CoupledTransientHeatTransferHydrogenTransport(
        heat_problem=my_heat_model,
        hydrogen_problem=my_model,
    )

    coupled_model.initialise()
    coupled_model.run()

The two problems can be defined on different meshes, in which case the temperature is interpolated onto the mesh of the hydrogen transport problem at each time step.

.. note::

    Coupled problems are not yet supported with :class:`festim.HydrogenTransportProblemDiscontinuous`.

----------------------
From a checkpoint file
----------------------

Temperature can also be read from a checkpoint file written by a previous simulation (see :ref:`Checkpoints`), with :func:`festim.read_function_from_file`:

.. code-block:: python

    my_model.temperature = F.read_function_from_file(
        filename="temperature.bp", name="f", timestamp=10.0, mesh=my_model.mesh.mesh
    )

.. note::

    The file must have been written by a :class:`festim.TemperatureExport` with :code:`format="checkpoint"`.
    The ``name`` argument is the name of the function in the file: for a :class:`festim.HeatTransferProblem` this is the name of its solution ``my_heat_model.u.name``, which is ``"f"`` unless you rename it before running.
    Files in the visualisation formats (``"vtx"``, ``"vtkhdf"``, ``"xdmf"``) cannot be read back.
