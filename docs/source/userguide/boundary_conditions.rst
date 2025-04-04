.. _boundary conditions:

===================
Boundary conditions
===================

The boundary conditions (BCs) are essential to FESTIM simulations. They describe the mathematical problem at the boundaries of the simulated domain.
If no BC is set on a boundary, it is assumed that the flux is null. This is also called a symmetry BC.

---------------
Basic BCs
---------------
These BCs can be used for heat transfer or hydrogen transport simulations.

Imposing the solution
----------------------

The value of solutions (concentration, temperature) can be imposed on boundaries with :class:`festim.FixedConcentrationBC` or :class:`festim.FixedTemperatureBC` classes respectively.

.. testcode:: BCs

    from festim import FixedConcentrationBC, FixedTemperatureBC, Species, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    my_hydrogen_bc = FixedConcentrationBC(subdomain=boundary, value=10, species=H)
    my_temperature_bc = FixedTemperatureBC(subdomain=boundary, value=10)

Both classes have two required arguments:

* :code:`subdomain`: the surface subdomain where the boundary condition is applied.
* :code:`value`: The value of the boundary condition. It can be a function of space and/or time (H/m3).

The ``subdomain`` argument can be a :class:`festim.Subdomain` object or a list of :class:`festim.Subdomain` objects.

However, as multiple species can be present in the same simulation, the ``species`` argument is required for the :class:`festim.FixedConcentrationBC` class. The ``species`` argument can be a single :class:`festim.Species` object or a list of :class:`festim.Species` objects.

Furthermore, for :class:`festim.FixedTemperatureBC`, the species can only be the temperature and therfore this arguement is not an option.

The ``value`` argument can be space and time dependent, and for concentrations, temperature dependent:

.. testcode:: BCs

    from festim import FixedConcentrationBC, FixedTemperatureBC, Species, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    my_custom_value = lambda x, t, T: 10 + x[0]**2 + t + T

    my_hydrogen_bc = FixedConcentrationBC(subdomain=boundary, value=my_custom_value, species=H)

.. note::

    When defining custom functions, only the arguments :code:`x`, :code:`t` and :code:`T` can be defined. 
    Where spatial coordinates x, y, z = use :code:`x[0]`, :code:`x[1]` and :code:`x[2]`. 
    Time dependence must use :code:`t`, and :code:`T` for temperature dependence.

Imposing the flux
------------------

When the flux needs to be imposed on a boundary, use the :class:`festim.ParticleFlux` or :class:`festim.HeatFluxBC` classes.

.. testcode:: BCs

    from festim import ParticleFluxBC, HeatFluxBC, Species, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    my_hydrogen_bc = ParticleFluxBC(subdomain=boundary, value=5, species=H)
    my_temperature_bc = HeatFluxBC(subdomain=boundary, value=5)


As for the Dirichlet boundary conditions, the flux can be dependent on space, time and temperature. 
But for fluxes, the values can also be dependent on a speices' concentration:

.. testcode:: BCs

    from festim import ParticleFluxBC, HeatFluxBC, Species, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")
    my_custom_value = lambda t, c: 10*t**2 + 2*c

    my_hydrogen_bc = ParticleFluxBC(subdomain=boundary, value=5, species=H, species_dependent_value={"c": H})

----------------------
Hydrogen transport BCs
----------------------

Some BCs are specific to hydrogen transport. FESTIM provides a handful of convenience classes making things a bit easier for the users.

Sievert's law of solubility
----------------------------

Impose the mobile concentration of hydrogen as :math:`c_\mathrm{m} = S(T) \sqrt{P}` where :math:`S` is the Sievert's solubility and :math:`P` is the partial pressure of hydrogen (see :class:`festim.SievertsBC`).

.. testcode:: BCs

    from festim import SievertsBC, SurfaceSubdomain, Species

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    custom_pressure_value = lambda t: 2 + t

    my_bc = SievertsBC(subdomain=3, S_0=2, E_S=0.1, species=H, pressure=custom_pressure_value)

Henry's law of solubility
--------------------------

Similarly, the mobile concentration can be set from Henry's law of solubility :math:`c_\mathrm{m} = K_H P` where :math:`K_H` is the Henry solubility (see :class:`festim.HenrysBC`).


.. testcode:: BCs

    from festim import HenrysBC, SurfaceSubdomain, Species

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    custom_pressure_value = lambda t: 5 * t

    my_bc = HenrysBC(subdomain=3, H_0=1.5, E_H=0.2, species=H, pressure=custom_pressure_value)

Surface reactions
------------------

Surface reactions on boundary can be defined with the :class:`festim.SurfaceReactionBC` class.

