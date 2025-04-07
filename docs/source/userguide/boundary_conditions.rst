.. _boundary conditions:

===================
Boundary conditions
===================

The boundary conditions (BCs) are essential to FESTIM simulations. They describe the mathematical problem at the boundaries of the simulated domain.
If no BC is set on a boundary, it is assumed that the flux is null. This is also called a symmetry BC.

----------------------
Heat transfer BCs
----------------------

Some BCs are specific to heat transfer. FESTIM provides a handful of convenience classes making things a bit easier for the users.

Imposing the temperature
---------------------------

The temperature can be imposed on boundaries with :class:`festim.FixedTemperatureBC`.

.. testcode:: BCs

    from festim import FixedTemperatureBC, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)

    my_bc = FixedTemperatureBC(subdomain=boundary, value=10)

The :class:`festim.FixedTemperatureBC` class has two required arguments:

* :code:`subdomain`: the surface subdomain where the fixed temperature is applied.
* :code:`value`: The value of the fixed temperature in units K.

The ``subdomain`` argument can be a :class:`festim.Subdomain` object or a list of :class:`festim.Subdomain` objects. 
The ``value`` argument can be a float, integer, function or a fenics object such as :class:`fem.Constant`, :class:`fem.Function`.

To define the temperature as space or time dependent, a function can be passed to the :code:`value` argument:

.. testcode:: BCs

    from festim import FixedTemperatureBC, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)

    my_custom_value = lambda x, t: 10 + x[0]**2 + t

    my_bc = FixedTemperatureBC(subdomain=boundary, value=my_custom_value)

.. note::

    When defining custom functions for values, only the arguments :code:`x`, :code:`t` and :code:`T` can be defined. 
    Where spatial coordinates x, y, z = use :code:`x[0]`, :code:`x[1]` and :code:`x[2]`.
    Time dependence must use :code:`t`, and :code:`T` for temperature dependence.

Imposing a heat flux
--------------------------

When a heat flux needs to be imposed on a boundary, use the :class:`festim.HeatFluxBC` class.

.. testcode:: BCs

    from festim import HeatFluxBC, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)

    my_flux_bc = HeatFluxBC(subdomain=boundary, value=5)

The :class:`festim.FixedTemperatureBC` class has two required arguments:

* :code:`subdomain`: the surface subdomain where the heat flux is applied.
* :code:`value`: The value of the heat flux in units W m\ :sup:`-2`.

As for the fixed temperature boundary condition, the flux can be dependent on space and time:

.. testcode:: BCs

    from festim import HeatFluxBC, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)

    my_custom_value = lambda x, t: 2 * x[0] + 10 * t

    my_flux_bc = HeatFluxBC(subdomain=boundary, value=my_custom_value)

----------------------
Hydrogen transport BCs
----------------------

Some BCs are specific to hydrogen transport. FESTIM provides a handful of convenience classes making things a bit easier for the users.


Imposing the concentration
---------------------------

The concentration of a defined species can be imposed on boundaries with :class:`festim.FixedConcentrationBC`.

.. testcode:: BCs

    from festim import FixedConcentrationBC, Species, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    my_bc = FixedConcentrationBC(subdomain=boundary, value=10, species=H)

The :class:`festim.FixedConcentrationBC` class has three required arguments:

* :code:`subdomain`: the surface subdomain where the fixed concentration is applied.
* :code:`value`: The value of the fixed concentration in units m\ :sup:`-3`.
* :code:`species`: The species for which the concentration is imposed.

The ``species`` argument can be a single :class:`festim.Species` object or a list of :class:`festim.Species` objects.

The imposed concentration can be dependent on space, time and temperature:

.. testcode:: BCs

    from festim import FixedConcentrationBC, Species, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    my_custom_value = lambda x, t, T: 10 + x[0]**2 + t + T

    my_bc = FixedConcentrationBC(subdomain=boundary, value=my_custom_value, species=H)


Imposing a particle flux
--------------------------

When a particle flux needs to be imposed on a boundary, use the :class:`festim.ParticleFlux` class.

.. testcode:: BCs

    from festim import ParticleFluxBC, Species, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    my_flux_bc = ParticleFluxBC(subdomain=boundary, value=2, species=H)

The :class:`festim.ParticleFlux` class has three required arguments:

* :code:`subdomain`: the surface subdomain where the particle flux is applied.
* :code:`value`: The value of the particle flux in units m\ :sup:`-2` s\ :sup:`-1`.
* :code:`species`: The species for which the particle flux is imposed.

As for the fixed concentration boundary condition, the flux can be dependent on space, time and temperature. 
But for particle fluxes, the values can also be dependent on a species' concentration:

.. testcode:: BCs

    from festim import ParticleFluxBC, Species, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    my_custom_value = lambda t, c: 10*t**2 + 2*c

    my_flux_bc = ParticleFluxBC(
        subdomain=boundary,
        value=my_custom_value,
        species=H,
        species_dependent_value={"c": H},
    )

.. note::

    The :code:`species_dependent_value` argument requires a dictionary to be passed, mapping any arguments in the custom function given to value, to any species defined.

    For instance with three species A, B and C, the dictionary can be defined as:
    
    .. testcode:: BCs

        from festim import Species

        A = Species(name="A")
        B = Species(name="B")
        C = Species(name="C")

        my_custom_value = lambda c_A, c_B, c_C: 2*c_A + 3*c_B + 4*c_C

        species_dependent_value = {"c_A": A, "c_B": B, "c_C": C}


Sievert's law of solubility
----------------------------

Impose the concentration of a species as :math:`c_\mathrm{m} = S(T) \sqrt{P}` where :math:`S` is the Sievert's solubility and :math:`P` is the partial pressure of the species on this surface (see :class:`festim.SievertsBC`).

.. testcode:: BCs

    from festim import SievertsBC, SurfaceSubdomain, Species

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    custom_pressure_value = lambda t: 2 + t

    my_bc = SievertsBC(subdomain=3, S_0=2, E_S=0.1, species=H, pressure=custom_pressure_value)

The :class:`festim.SievertsBC` class has five required arguments:

* :code:`subdomain`: the surface subdomain where the concentration is applied.
* :code:`S_0`: The pre-exponential factor for the solubility of the contact material in m\ :sup:`-3` Pa\ :sup:`-0.5`.
* :code:`E_S`: The activation energy for the solubility of the contact material in eV.
* :code:`species`: The species for which the particle flux is imposed.
* :code:`pressure`: The gas pressure in Pa.


Henry's law of solubility
--------------------------

Similarly, the the concentration of a species can be set from Henry's law of solubility :math:`c_\mathrm{m} = K_H P` where :math:`K_H` is the Henry solubility (see :class:`festim.HenrysBC`).

.. testcode:: BCs

    from festim import HenrysBC, SurfaceSubdomain, Species

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    pressure_value = lambda t: 5 * t

    my_bc = HenrysBC(subdomain=3, H_0=1.5, E_H=0.2, species=H, pressure=pressure_value)

The :class:`festim.HenrysBC` class has five required arguments:

* :code:`subdomain`: the surface subdomain where the concentration is applied.
* :code:`H_0`: The pre-exponential factor for the solubility of the contact material in m\ :sup:`-3` Pa\ :sup:`-1`.
* :code:`E_H`: The activation energy for the solubility of the contact material in eV.
* :code:`species`: The species for which the particle flux is imposed.
* :code:`pressure`: The gas pressure in Pa.


Surface reactions
------------------

Surface reactions on boundary can be defined with the :class:`festim.SurfaceReactionBC` class.

The surface reaction class can be used to impose dissociation and recombination reactions on the surface of the material.

.. testcode:: BCs

    from festim import Species, SurfaceReactionBC, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    my_bc = SurfaceReactionBC(
        reactant=[H],
        gas_pressure=1e5,
        k_r0=1,
        E_kr=0.1,
        k_d0=1e-5,
        E_kd=0.1,
        subdomain=boundary,
    )

The :class:`festim.SurfaceReactionBC` class has the following required arguments:

* :code:`reactant`: The species that is involved in the reaction.
* :code:`gas_pressure`: The gas pressure in Pa.
* :code:`k_r0`: The pre-exponential factor for the reaction rate in ms\ :sup:`-1` or m\ :sup:`4` s\ :sup:`-1`.
* :code:`E_kr`: The activation energy for the reaction rate in eV.
* :code:`kd_0`: The pre-exponential factor for the desorption rate in m\ :sup:`-2` s\ :sup:`-1` Pa\ :sup:`-1`.
* :code:`E_kd`: The activation energy for the desorption rate in eV.
* :code:`subdomain`: The subdomain where the reaction is applied.
