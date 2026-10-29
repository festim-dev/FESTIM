.. _sources_guide:

=======
Sources
=======

.. testsetup:: sources

    import festim as F

    my_mat = F.Material(D_0=1, E_D=0.1)
    my_vol = F.VolumeSubdomain(id=1, material=my_mat)
    H = F.Species(name="H")

Sources describe the production (or removal) of particles or heat inside the simulated domain, as opposed to :ref:`boundary conditions <boundary_conditions>` which act on its surfaces.

All sources in FESTIM require a volume subdomain, defined with the :class:`festim.VolumeSubdomain` class.
See the :ref:`Volume Subdomains` section for more information on how to define subdomains.

Sources are passed to the problem as a list in its ``sources`` attribute.

.. testcode:: sources

    my_model = F.HydrogenTransportProblem()

    my_model.sources = [
        F.ParticleSource(value=1e20, volume=my_vol, species=H),
    ]

-----------------
Particle sources
-----------------

Volumetric sources of a species are set with the :class:`festim.ParticleSource` class.
A source is defined by its value (in :math:`\mathrm{m}^{-3}\,\mathrm{s}^{-1}`), the volume subdomain where it is applied and the species it applies to.

.. testcode:: sources

    from festim import ParticleSource

    my_source = ParticleSource(value=1e20, volume=my_vol, species=H)

The value can be dependent on space, time and temperature:

.. testcode:: sources

    from festim import ParticleSource

    my_custom_value = lambda x, t, T: 1e20 * x[0] + 1e18 * t + T

    my_source = ParticleSource(value=my_custom_value, volume=my_vol, species=H)

.. note::

    When defining custom functions for values, only the arguments :code:`x`, :code:`t` and :code:`T` can be defined (plus the species concentrations described below).
    Spatial coordinates can be referred to by their indices, such as :code:`x[0]`, :code:`x[1]`, and :code:`x[2]`, regardless of the coordinate system used.
    Time dependence must use :code:`t`, and :code:`T` for temperature dependence.

Species-dependent sources
--------------------------

The value of a particle source can also depend on the concentration of one or several species.
As for :class:`festim.ParticleFluxBC`, the :code:`species_dependent_value` argument maps the names of the arguments of the custom function to the corresponding :class:`festim.Species` objects:

.. testcode:: sources

    from festim import ParticleSource, Species

    A = Species(name="A")
    B = Species(name="B")

    my_custom_value = lambda c_A, c_B: 2 * c_A - 3 * c_B

    my_source = ParticleSource(
        value=my_custom_value,
        volume=my_vol,
        species=A,
        species_dependent_value={"c_A": A, "c_B": B},
    )

This is the mechanism used internally by FESTIM to expand :ref:`reactions <reactions_guide>` into sources.
For reactions between species (trapping, radioactive decay, hydride formation...), prefer the dedicated reaction classes described in the :ref:`Species <species_user_guide>` page.

Implantation flux
-----------------

Hydrogen implanted in a material can be simulated by a Gaussian-shaped volumetric source.
Unlike FESTIM 1.x, there is no dedicated class for it in FESTIM 2: the profile is simply given as a function of space.
For an implantation flux :math:`\varphi` (in :math:`\mathrm{m}^{-2}\,\mathrm{s}^{-1}`), an implantation depth :math:`R_p` and a width :math:`\sigma`:

.. math::

    S(x) = \frac{\varphi}{\sigma \sqrt{2 \pi}} \exp \left( - \frac{(x - R_p)^2}{2 \sigma^2} \right)

.. testcode:: sources

    import numpy as np
    import ufl
    from festim import ParticleSource

    flux = 1e20  # H/m2/s
    imp_depth = 1e-9  # m
    width = 1e-9  # m

    def gaussian(x):
        return (
            flux
            / (width * np.sqrt(2 * np.pi))
            * ufl.exp(-0.5 * ((x[0] - imp_depth) / width) ** 2)
        )

    my_source = ParticleSource(value=gaussian, volume=my_vol, species=H)

.. note::

    The function above is evaluated symbolically on the mesh coordinates, so the mathematical functions must come from ``ufl`` (:code:`ufl.exp`, :code:`ufl.sin`...) rather than from ``numpy`` or ``math``.
    Plain arithmetic operations (``+``, ``*``, ``**``...) and numpy constants are fine.

The implantation flux can be made time dependent by adding :code:`t` to the arguments of the function.

-------------
Heat sources
-------------

Volumetric heat sources (in :math:`\mathrm{W}\,\mathrm{m}^{-3}`) are set with the :class:`festim.HeatSource` class and are passed to a :class:`festim.HeatTransferProblem` (see :ref:`Temperature <temperature_guide>`).

.. testcode:: sources

    from festim import HeatSource

    my_heat_source = HeatSource(value=1e6, volume=my_vol)

As for particle sources, the value can be dependent on space and time:

.. testcode:: sources

    from festim import HeatSource

    my_custom_value = lambda x, t: 1e6 * x[0] + 1e4 * t

    my_heat_source = HeatSource(value=my_custom_value, volume=my_vol)

.. note::

    Heat sources cannot be temperature dependent.

------------------
Radioactive decay
------------------

In FESTIM 1.x, radioactive decay was a source (``festim.RadioactiveDecay``).
In FESTIM 2 it is a reaction, defined with the :class:`festim.DecayReaction` class and passed to the ``reactions`` attribute of the problem (see :ref:`Reactions <reactions_guide>`).

.. testcode:: sources

    from festim import DecayReaction, Species

    tritium = Species(name="T")

    my_model.species = [tritium]
    my_model.reactions = [
        DecayReaction(reactant=tritium, half_life=3.888e8, volume=my_vol),
    ]
