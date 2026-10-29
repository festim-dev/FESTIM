.. _drift_guide:

===========
Drift terms
===========

A drift term makes hydrogen move by something other than its own concentration
gradient. All of them add a velocity to the flux, :math:`J = -D \nabla c + c \mathbf{v}`,
and differ only in what sets :math:`\mathbf{v}`. See the :doc:`theory guide </theory>`
for the equations.

They are passed to the problem as ``drift_terms``.

Soret effect
------------

Thermodiffusion along a temperature gradient, with the heat of transport
:math:`Q^*` in eV:

.. code-block:: python

    import festim as F

    my_model.drift_terms = [
        F.SoretTerm(species=mobile_H, Q_star=0.2, subdomain=my_volume)
    ]

The term needs a temperature that varies in space, so give it as a function of ``x`` or
couple an instance of :class:`festim.CoupledTransientHeatTransferHydrogenTransport`. A uniform
temperature gradient, and then the Soret has no effect. FESTIM warns when this happens. 

Electromigration
----------------

Drift of a charged species in an electric potential, in volts. ``charge`` is the charge
number :math:`z`:

.. code-block:: python

    my_model.drift_terms = [
        F.ElectromigrationTerm(
            species=hydroxyl,
            charge=1,
            potential=lambda x: 0.5 * (1 - x[0] / 5e-4),
            subdomain=membrane,
        )
    ]

FESTIM does not solve for the potential, so it must be passed in as an existing field. Give it as a float, a
callable of ``x``, ``t`` and/or ``T``, or a fenics object. As with the Soret term, a spatially uniform potential
makes the term do nothing, and FESTIM sends a warning.

Advection
---------

Hydrogen carried by a moving fluid, with the velocity given directly as a
``dolfinx.fem.Function`` on a vector function space. 

.. code-block:: python

    my_model.drift_terms = [
        F.AdvectionTerm(velocity=my_velocity_field, subdomain=my_volume, species=H)
    ]

.. _conservative_form:

The divergence form
-------------------

Every drift term is assembled as :math:`\nabla \cdot (c\mathbf{v})`. There is no option
to assemble :math:`\mathbf{v} \cdot \nabla c` instead: the two differ by
:math:`c \nabla \cdot \mathbf{v}`. Only the divergence form conserves the species.

The boundary term the divergence form leaves behind is the
natural boundary condition, so **flux boundary conditions constrain the total flux**,
drift included, and a boundary with no condition on it behaves like an impermeable wall (zero total flux, with
the drift balanced by back-diffusion). If the boundary is intended to be an outlet, the `OutflowBC` is needed
(see :ref:`outflow`).

.. note::

    FESTIM does not stabilise the advection-diffusion form. When drift dominates, i.e. when the cell Péclet number :math:`\mathrm{Pe} = |\mathbf{v}| h / D` becomes large, the solution can oscillate. Refine the mesh where the drift dominates to mitigate this issue.


.. warning::

    **This changes results for existing advection models.** Before FESTIM 2.2,
    :class:`festim.AdvectionTerm` used :math:`\mathbf{v} \cdot \nabla c`. Your results
    are unchanged if both of the following hold:

    * the velocity field is divergence-free, as an incompressible flow is;
    * every boundary the flow crosses carries a boundary condition.

    If the flow leaves through a boundary you did not tag, that boundary is now a closed
    end and the species backs up against it. Add a :class:`festim.OutflowBC` there.

.. _outflow:

Letting the species out
-----------------------

:class:`festim.OutflowBC` marks a surface the flow leaves through. It cancels the drift
boundary term, so the natural condition there becomes zero *diffusive* flux, and the species is carried out at
the rate the drift delivers it:

.. code-block:: python

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=inlet, value=1.0, species=H),
        F.OutflowBC(subdomain=outlet, species=H),
    ]

On a codimensional problem the surface may be the boundary of a manifold, e.g. the outlet of
a 1D fluid running along a pipe wall.

:class:`festim.OutflowBC` does nothing on a surface where no drift acts on the species.

Surface fluxes
--------------

:class:`festim.SurfaceFlux` reports the total flux
:math:`(-D \nabla c + c\mathbf{v}) \cdot \mathbf{n}`, so drift contributes to it.