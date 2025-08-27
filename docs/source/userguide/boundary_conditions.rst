.. _boundary_conditions:

===================
Boundary conditions
===================

The boundary conditions (BCs) are essential to FESTIM simulations. They describe the mathematical problem at the boundaries of the simulated domain.
If no BC is set on a boundary, it is assumed that the flux is null. This is also called a symmetry BC.

All boundary conditions in FESTIM require a surface subdomain. 
The subdomain can be a surface, an edge or a point and can be defined with the :class:`festim.SurfaceSubdomain` class.
See the :ref:`Surface Subdomains` section for more information on how to define subdomains.

-----------------------
Hydrogen transport BCs
-----------------------

Some BCs are specific to hydrogen transport. FESTIM provides a handful of convenience classes making things a bit easier for the users.


Imposing the concentration
---------------------------

The concentration of a defined species can be imposed on boundaries with :class:`festim.FixedConcentrationBC`.

.. testcode:: BCs

    from festim import FixedConcentrationBC, Species, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    my_bc = FixedConcentrationBC(subdomain=boundary, value=10, species=H)

The imposed concentration can be dependent on space, time and temperature:

.. testcode:: BCs

    from festim import FixedConcentrationBC, Species, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    my_custom_value = lambda x, t, T: 10 + x[0]**2 + t + T

    my_bc = FixedConcentrationBC(subdomain=boundary, value=my_custom_value, species=H)

.. note::

    When defining custom functions for values, only the arguments :code:`x`, :code:`t` and :code:`T` can be defined. 
    Spatial coordinates can be referred to by their indices, such as :code:`x[0]`, :code:`x[1]`, and :code:`x[2]`, regardless of the coordinate system used.
    Time dependence must use :code:`t`, and :code:`T` for temperature dependence.

Imposing a particle flux
--------------------------

When a particle flux needs to be imposed on a boundary, use the :class:`festim.ParticleFluxBC` class.

.. testcode:: BCs

    from festim import ParticleFluxBC, Species, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    my_flux_bc = ParticleFluxBC(subdomain=boundary, value=2, species=H)

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


Henry's law of solubility
--------------------------

Similarly, the the concentration of a species can be set from Henry's law of solubility :math:`c_\mathrm{m} = K_H P` where :math:`K_H` is the Henry solubility (see :class:`festim.HenrysBC`).

.. testcode:: BCs

    from festim import HenrysBC, SurfaceSubdomain, Species

    boundary = SurfaceSubdomain(id=1)
    H = Species(name="Hydrogen")

    pressure_value = lambda t: 5 * t

    my_bc = HenrysBC(subdomain=3, H_0=1.5, E_H=0.2, species=H, pressure=pressure_value)


Surface reactions
------------------

Surface reactions on boundary can be defined with the :class:`festim.SurfaceReactionBC` class.

The surface reaction class can be used to impose dissociation and recombination reactions on the surface of the material.
A reaction is defined by specifying the reactants, products, and forward/backward rate constants. For example:


.. testcode:: BCs

    from festim import Species, SurfaceReactionBC, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    A = Species("A")
    B = Species("B")
    C = Species("C")

    my_bc = SurfaceReactionBC(
        reactant=[A, B],
        gas_pressure=1e5,
        k_r0=1,
        E_kr=0.1,
        k_d0=1e-5,
        E_kd=0.1,
        subdomain=boundary,
    )

The net reaction rate is:

.. math::
    R = K_r c_A c_B - K_d c_C

where :math:`K_r` and :math:`K_d` are the temperature-dependent forward and backward rate constants, respectively, and :math:`c_A`, :math:`c_B`, and :math:`c_C` are the concentrations of species A, B, and C at the surface.
From this FESTIM automatically applies the corresponding fluxes as Neumann boundary conditions (see :class:`festim.ParticleFluxBC`).

Recombination and Dissociation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hydrogen recombination/dissociation can be modelled as:

.. math::
    \mathrm{H + H} \overset{K_r}{\underset{K_d}{\rightleftharpoons}} \mathrm{H_2}

If the partial pressure of :math:`\mathrm{H}_2` is known or assumed constant, the product species does not need to be included explicitly, and the flux simplifies to:

.. math::
    \mathbf{J}_{\mathrm{H}} \cdot \mathbf{n} = K_r c_{\mathrm{H}}^2 - K_d P_{\mathrm{H_2}}

Both rate coefficients can follow Arrhenius laws.

.. testcode:: BCs

    from festim import Species, SurfaceReactionBC, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species("H")

    my_bc = SurfaceReactionBC(
        reactant=[H, H],
        gas_pressure=1e5,
        k_r0=1,
        E_kr=0.1,
        k_d0=1e-5,
        E_kd=0.1,
        subdomain=boundary,
    )

Multiple isoptopes
~~~~~~~~~~~~~~~~~~~

Surface reactions can involve multiple hydrogen isotopes, enabling the modelling of more complex interactions between species. 
For example, in a system with both mobile hydrogen and tritium, various molecular recombination pathways may occur at the surface, resulting in the formation of :math:`\mathrm{H_2}`, :math:`\mathrm{T_2}`, and :math:`\mathrm{HT}`:

.. math::
    \mathrm{H + H} \rightleftharpoons \mathrm{H_2}, \quad
    \mathrm{T + T} \rightleftharpoons \mathrm{T_2}, \quad
    \mathrm{H + T} \rightleftharpoons \mathrm{HT}

.. testcode:: BCs

    from festim import Species, SurfaceReactionBC, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)
    H = Species("H")
    T = Species("T")

    reac1 = SurfaceReactionBC(
        reactant=[H, H],
        gas_pressure=1e6,
        k_r0=1.0,
        E_kr=0.1,
        k_d0=0.5,
        E_kd=0.1,
        subdomain=boundary,
    )
    reac2 = SurfaceReactionBC(
        reactant=[T, T],
        gas_pressure=1e6,
        k_r0=1.0,
        E_kr=0.1,
        k_d0=0.5,
        E_kd=0.1,
        subdomain=boundary,
    )
    reac3 = SurfaceReactionBC(
        reactant=[H, T],
        gas_pressure=1e6,
        k_r0=1.0,
        E_kr=0.1,
        k_d0=0.5,
        E_kd=0.1,
        subdomain=boundary,
    )


Exchange Reactions
~~~~~~~~~~~~~~~~~~~

Certain isotopic exchange processes can also be approximated, e.g.:

.. math::
    \mathrm{T + H_2} \rightleftharpoons \mathrm{H} + \mathrm{HT}

If the :math:`\mathrm{H}_2` concentration is assumed much larger than :math:`\mathrm{HT}`, this reduces to a first-order process in :math:`\mathrm{T}`. 
Such fluxes can be implemented using :class:`festim.ParticleFluxBC` with user-defined expressions.

.. testcode:: BCs

    import ufl
    from festim import (
        Material,
        ParticleFluxBC,
        Species,
        SurfaceSubdomain,
        VolumeSubdomain,
        k_B,
    )

    boundary = SurfaceSubdomain(id=1)
    my_mat = Material(D_0=1, E_D=0.1)
    volume = VolumeSubdomain(id=1, material=my_mat)
    tritium = Species("tritium")
    Kr_0 = 1.0
    E_Kr = 0.1


    def my_custom_recombination_flux(c, T):
        Kr_0_custom = 1.0
        E_Kr_custom = 0.5  # eV
        h2_conc = 1e25  # assumed constant H2 concentration in

        recombination_flux = (
            -(Kr_0 * ufl.exp(-E_Kr / (k_B * T))) * c**2
            - (Kr_0_custom * ufl.exp(-E_Kr_custom / (k_B * T))) * h2_conc * c
        )
        return recombination_flux


    my_custom_flux = ParticleFluxBC(
        value=my_custom_recombination_flux,
        subdomain=boundary,
        species_dependent_value={"c": tritium},
        species=tritium,
    )


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


To define the temperature as space or time dependent, a function can be passed to the :code:`value` argument:

.. testcode:: BCs

    from festim import FixedTemperatureBC, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)

    my_custom_value = lambda x, t: 10 + x[0]**2 + t

    my_bc = FixedTemperatureBC(subdomain=boundary, value=my_custom_value)


Imposing a heat flux
--------------------------

When a heat flux needs to be imposed on a boundary, use the :class:`festim.HeatFluxBC` class.

.. testcode:: BCs

    from festim import HeatFluxBC, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)

    my_flux_bc = HeatFluxBC(subdomain=boundary, value=5)


As for the fixed temperature boundary condition, the flux can be dependent on space and time.
But for heat fluxes, the values can also be dependent on a temperature:

.. testcode:: BCs

    from festim import HeatFluxBC, SurfaceSubdomain

    boundary = SurfaceSubdomain(id=1)

    my_custom_value = lambda x, t, T: 2 * x[0] + 10 * t + T

    my_flux_bc = HeatFluxBC(subdomain=boundary, value=my_custom_value)