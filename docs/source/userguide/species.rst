.. _species_user_guide:

======================
Species & Reactions
======================

.. testsetup:: species

    import festim as F

    my_mat = F.Material(D_0=1, E_D=0.1)
    my_vol = F.VolumeSubdomain(id=1, material=my_mat)

In FESTIM 2, every concentration field solved for is a *species*.
A hydrogen transport problem has at least one mobile species (for instance mobile hydrogen), and can have as many additional species as needed: trapped hydrogen, several isotopes, helium, vacancies...
Species interact through *reactions*, which is how trapping and detrapping, radioactive decay or hydride formation are modelled.

Species are passed to the problem as a list in its ``species`` attribute, and reactions in its ``reactions`` attribute.

.. testcode:: species

    my_model = F.HydrogenTransportProblem()

    mobile_H = F.Species(name="H")
    trapped_H = F.Species(name="trapped_H", mobile=False)

    my_model.species = [mobile_H, trapped_H]

--------
Species
--------

Species are defined with the :class:`festim.Species` class.

.. testcode:: species

    from festim import Species

    H = Species(name="H")

The ``name`` is used to identify the species in exports and error messages.
By default, a species is mobile: it diffuses with the diffusivity of the material it is in (see :doc:`Subdomains & Materials <subdomains>`).
A species that is not mobile (typically a trapped concentration) is defined with ``mobile=False``:

.. testcode:: species

    from festim import Species

    trapped_H = Species(name="trapped_H", mobile=False)

Immobile species do not diffuse and only evolve through reactions and sources.

In multi-material problems using :class:`festim.HydrogenTransportProblemDiscontinuous`, a species can be restricted to some of the volume subdomains with the ``subdomains`` argument:

.. testcode:: species

    from festim import Species

    H = Species(name="H", subdomains=[my_vol])

.. _implicit_species:

Implicit species
----------------

Some concentrations are not solved for but are computed from other species.
The most common example is the concentration of *empty* trap sites :math:`n - c_\mathrm{t}` where :math:`n` is the trap density and :math:`c_\mathrm{t}` the concentration of trapped hydrogen.
These are defined with the :class:`festim.ImplicitSpecies` class: its concentration is ``n`` minus the concentrations of the species listed in ``others``.

.. testcode:: species

    from festim import Species, ImplicitSpecies

    trapped_H = Species(name="trapped_H", mobile=False)

    empty_traps = ImplicitSpecies(n=1e25, others=[trapped_H], name="empty_traps")

The density ``n`` can be a function of space and time:

.. testcode:: species

    from festim import Species, ImplicitSpecies

    trapped_H = Species(name="trapped_H", mobile=False)

    empty_traps = ImplicitSpecies(
        n=lambda x, t: 1e25 * x[0] + 1e20 * t,
        others=[trapped_H],
        name="empty_traps",
    )

Implicit species have no governing equation: they are not passed to ``my_model.species`` but are used as reactants in reactions.

.. _reactions_guide:

----------
Reactions
----------

A reaction couples species together. It does not enter the formulation directly: it is expanded into volumetric :class:`festim.ParticleSource` objects, each reactant getting a sink and each product a source (see :ref:`Sources <sources_guide>`).

Reactions are defined by their reactants, their products and their rate. FESTIM provides several reaction classes.

Arrhenius reactions
-------------------

:class:`festim.ArrheniusReaction` is a reaction whose forward and backward rate coefficients follow Arrhenius laws:

.. math::

    R = k_0 \exp \left(-\frac{E_k}{k_B T}\right) \prod_i c_i - p_0 \exp \left(-\frac{E_p}{k_B T}\right) \prod_j c_j

where :math:`c_i` and :math:`c_j` are the concentrations of the reactants and of the products, respectively.
This is typically used to model trapping and detrapping:

.. testcode:: species

    from festim import Species, ImplicitSpecies, ArrheniusReaction

    mobile_H = Species(name="H")
    trapped_H = Species(name="trapped_H", mobile=False)
    empty_traps = ImplicitSpecies(n=1e25, others=[trapped_H], name="empty_traps")

    trapping = ArrheniusReaction(
        reactant=[mobile_H, empty_traps],
        product=trapped_H,
        k_0=1e-16,
        E_k=0.2,
        p_0=1e13,
        E_p=1.0,
        volume=my_vol,
    )

    my_model.species = [mobile_H, trapped_H]
    my_model.reactions = [trapping]

Here, mobile hydrogen reacts with an empty trap site to give trapped hydrogen, which is the McNabb & Foster model (see :doc:`theory guide </theory>`).

A reaction is irreversible when no product is given. ``p_0`` and ``E_p`` must then be omitted.

Reactions can be restricted to some volume subdomains: when a reaction should take place in several subdomains, define one reaction per subdomain.

.. note::

    A species listed twice as a reactant appears squared in the rate and is consumed twice as fast. For example, the reaction :math:`2 \mathrm{H} \rightarrow \mathrm{H}_2` is written with ``reactant=[H, H]``.

Generic reactions
-----------------

:class:`festim.GenericReaction` follows the same mass-action law but takes arbitrary rate coefficients instead of Arrhenius parameters.
The forward and backward rates can be constants, or functions of the temperature ``T``, of the time ``t`` and of the space coordinates ``x``:

.. testcode:: species

    from festim import Species, GenericReaction

    A = Species(name="A")
    B = Species(name="B")
    C = Species(name="C")

    my_reaction = GenericReaction(
        reactant=[A, B],
        product=C,
        forward_rate=lambda T: 1e-3 * T,
        backward_rate=2.0,
        volume=my_vol,
    )

The rates can also depend on the concentration of other species, through the ``arg_to_species`` argument which maps the argument names of the callable to :class:`festim.Species` objects:

.. testcode:: species

    from festim import Species, GenericReaction

    A = Species(name="A")
    B = Species(name="B")

    my_reaction = GenericReaction(
        reactant=A,
        product=B,
        forward_rate=lambda c_B: 1e-3 * (1 + c_B),
        backward_rate=None,
        volume=my_vol,
        arg_to_species={"c_B": B},
    )

Arbitrary rates
---------------

When the rate does not follow a mass-action law at all, use :class:`festim.ReactionBase` and give the net rate :math:`R` directly.
Each reactant is consumed at rate :math:`R` and each product is produced at rate :math:`R`:

.. testcode:: species

    from festim import Species, ReactionBase

    A = Species(name="A")
    B = Species(name="B")

    my_reaction = ReactionBase(
        reaction_rate=lambda c_A, c_B: 2.0 * (c_A - c_B),
        reactant=A,
        product=B,
        volume=my_vol,
        arg_to_species={"c_A": A, "c_B": B},
    )

Radioactive decay
-----------------

Radioactive decay is a first-order reaction defined with :class:`festim.DecayReaction`.
The decay constant :math:`\lambda = \ln(2) / t_{1/2}` is built from the ``half_life`` (in seconds).
The decay product can be tracked by giving it as ``product``:

.. testcode:: species

    from festim import Species, DecayReaction

    T = Species(name="T")  # tritium
    He = Species(name="He")  # helium-3

    decay = DecayReaction(
        reactant=T,
        half_life=3.888e8,  # ~12.3 years
        volume=my_vol,
        product=He,
    )

For a trapped species to decay as well, define one :class:`festim.DecayReaction` per species.

------
Traps
------

Defining a trap requires a trapped species, an implicit species for the empty sites and an Arrhenius reaction.
For the common case of one mobile species and one trapping level, the :class:`festim.Trap` convenience class does all of this.
Traps are passed to the ``traps`` attribute of the problem, and the trapped species and the reaction are created automatically when the model is initialised:

.. testcode:: species

    from festim import Species, Trap

    mobile_H = Species(name="H")

    my_trap = Trap(
        name="trapped_H",
        mobile_species=mobile_H,
        k_0=1e-16,
        E_k=0.2,
        p_0=1e13,
        E_p=1.0,
        n=1e25,
        volume=my_vol,
    )

    my_model.species = [mobile_H]
    my_model.traps = [my_trap]

This is strictly equivalent to the explicit definition of the trapped species, empty sites and :class:`festim.ArrheniusReaction` shown above.
A trap in several materials needs one :class:`festim.Trap` per volume subdomain.
The trap density ``n`` can be a function of space and time, exactly like the ``n`` of an :ref:`implicit species <implicit_species>`.

.. note::

    In FESTIM 1.x, ``Trap`` objects took a ``density`` and a list of ``materials``, and extrinsic traps had their own class.
    In FESTIM 2, a time-dependent trap density is given as a function of ``t`` and a trap whose density is itself solved for is written as an immobile :class:`festim.Species` with its own reactions and sources.
