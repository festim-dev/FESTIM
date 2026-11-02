==========
Interfaces
==========

In a :class:`festim.HydrogenTransportProblemDiscontinuous`, each volume subdomain has its own concentration field, so the concentration can jump where two subdomains meet.
An interface defines how the fields on either side are coupled.

Every interface has an ``id``, the tag of its facets in the facet meshtags, and the two volume ``subdomains`` it separates.
The order of ``subdomains`` matters: the first subdomain is side 0 and the second is side 1.

FESTIM provides two kinds of interface conditions:

* :class:`festim.Interface` assumes local thermodynamic equilibrium (LTE) and links the two interfacial concentrations with an algebraic relation.
* :class:`festim.InterfaceReaction` replaces this relation with a reversible reaction of finite rate. LTE is recovered as the fast-kinetics limit.

.. testsetup:: interfaces

    from festim import Material, VolumeSubdomain1D

    mat = Material(D_0=1, E_D=0)
    vol1 = VolumeSubdomain1D(id=1, material=mat, borders=[0, 0.5])
    vol2 = VolumeSubdomain1D(id=2, material=mat, borders=[0.5, 1])

Local thermodynamic equilibrium
===============================

At equilibrium, the chemical potential of hydrogen is continuous across the interface.
The concentration is not: its relation to the partial pressure :math:`P` depends on how hydrogen dissolves on each side.

Choosing Sievert's or Henry's law
---------------------------------

The solubility law of each material is set with the ``solubility_law`` argument of :class:`festim.Material` (see :ref:`Materials`):

* ``"sievert"``: Sievert's law, :math:`c = K_S \sqrt{P}`. Hydrogen molecules dissociate and dissolve as atoms. This is the case in metals.
* ``"henry"``: Henry's law, :math:`c = K_H P`. Hydrogen dissolves as molecules. This is the case in liquids such as molten salts.

The solubility is defined by ``K_S_0`` and ``E_K_S`` in both cases, but its units depend on the law: :math:`\mathrm{m}^{-3}\,\mathrm{Pa}^{-1/2}` for Sievert's law and :math:`\mathrm{m}^{-3}\,\mathrm{Pa}^{-1}` for Henry's law.

.. testcode:: interfaces

    from festim import Material

    metal = Material(D_0=1, E_D=0, K_S_0=2, E_K_S=0, solubility_law="sievert")
    salt = Material(D_0=1, E_D=0, K_S_0=3, E_K_S=0, solubility_law="henry")

In multi-material problems, every material must have a solubility law. The default, ``"none"``, only works for single-material problems.

.. warning::

    A solubility value is only meaningful together with the law used to obtain it: a Sievert's solubility cannot be converted into a Henry's solubility with a constant factor.
    When taking a solubility from the literature, check which law (and which dissolved species, atoms or molecules) the authors assumed.

FESTIM chooses the interface condition from the solubility laws of the two materials.
Between two Sievert (or two Henry) materials:

.. math::

    \frac{c_0}{K_{S,0}} = \frac{c_1}{K_{S,1}}

Between a Sievert material (side 0) and a Henry material (side 1):

.. math::

    c_1 = K_{H,1} \left( \frac{c_0}{K_{S,0}} \right)^2

Imposing the equilibrium
------------------------

:class:`festim.Interface` imposes the equilibrium weakly, either with a penalty method (default) or with Nitsche's method:

.. testcode:: interfaces

    from festim import Interface, InterfaceMethod, Species

    H = Species("H", subdomains=[vol1, vol2])

    interface = Interface(id=3, subdomains=[vol1, vol2], penalty_term=100)
    interface_nitsche = Interface(id=3, subdomains=[vol1, vol2], method=InterfaceMethod.nitsche)

The condition applies to every species defined in both subdomains.

Limitations of local equilibrium
--------------------------------

The LTE condition is algebraic: it fixes a ratio, or a power law, between the two interfacial concentrations, whatever the flux crossing the interface.
It relies on three assumptions:

1. The interface equilibrates quickly compared to bulk transport. The interface then has no timescale of its own and never limits the transport.
2. A single exchange pathway connects the two sides.
3. The species carrying hydrogen on each side is known in advance, so that a single solubility law applies.

An interface carrying a net flux is never exactly at equilibrium, so LTE is always an approximation there.
When one of these assumptions is questionable (slow interface, several competing pathways, several isotopes forming mixed molecules such as HT), use interface reactions instead.

Interface reactions
===================

:class:`festim.InterfaceReaction` models the exchange across the interface as a reversible reaction between reactants on side 0 and products on side 1.
The reaction rate follows the law of mass action:

.. math::

    R = k_+ \prod_i c_{0,i} - k_- \prod_j c_{1,j}

where :math:`c_{0,i}` are the concentrations of the reactants on side 0 and :math:`c_{1,j}` are the concentrations of the products on side 1.
:math:`R` is a flux (:math:`\mathrm{m}^{-2}\,\mathrm{s}^{-1}`): each reactant leaves side 0 and each product enters side 1 at the rate :math:`R`.
The units of ``k_plus`` and ``k_minus`` therefore depend on the number of reactants and products.

Every reactant must be defined in the first subdomain and every product in the second subdomain, otherwise a ``ValueError`` is raised.

.. note::

    A species listed twice as a reactant appears squared in the rate and is consumed twice as fast.
    For example, the reaction :math:`2 \mathrm{H} \rightleftharpoons \mathrm{H}_2` is written with ``reactants=[H, H]``.

Detailed balance
----------------

``k_plus`` and ``k_minus`` are not independent.
At equilibrium (:math:`R = 0`), the reaction must give the same condition as LTE, which fixes their ratio:

.. math::

    \frac{k_+}{k_-} = K

where :math:`K` is the equilibrium constant of the reaction, obtained from the same solubilities as the LTE condition.
Thermodynamics only fixes the ratio of the rate constants.
Their magnitude, which sets how fast the interface equilibrates, is an additional input that must come from atomistic modelling or experiments.

.. note::

    FESTIM does not check that :math:`k_+/k_-` satisfies detailed balance: this is the responsibility of the user.
    The rate constants are constant values, with no temperature dependence, and must be evaluated at the temperature of the simulation.

First-order exchange
--------------------

The simplest reaction, :math:`\mathrm{H}_0 \rightleftharpoons \mathrm{H}_1`, transfers hydrogen across the interface without changing its form, for example at a metal/metal interface:

.. math::

    R = k_+ c_0 - k_- c_1, \qquad \frac{k_+}{k_-} = \frac{K_{S,1}}{K_{S,0}}

The same species can be used as reactant and product:

.. testcode:: interfaces

    from festim import InterfaceReaction, Species

    H = Species("H", subdomains=[vol1, vol2])

    K_S_0, K_S_1 = 2, 4
    k_plus = 1e3
    k_minus = k_plus * K_S_0 / K_S_1  # detailed balance

    interface = InterfaceReaction(
        id=3,
        subdomains=[vol1, vol2],
        k_plus=k_plus,
        k_minus=k_minus,
        reactants=[H],
        products=[H],
    )

Recombination
-------------

At a metal/liquid interface, hydrogen atoms leaving the metal can recombine into molecules that dissolve in the liquid, :math:`2 \mathrm{H} \rightleftharpoons \mathrm{H}_2`:

.. math::

    R = k_+ c_\mathrm{H}^2 - k_- c_{\mathrm{H}_2}, \qquad \frac{k_+}{k_-} = \frac{K_H}{K_S^2}

where :math:`K_S` is the Sievert's solubility of the metal and :math:`K_H` the Henry's solubility of the liquid.

.. testcode:: interfaces

    from festim import InterfaceReaction, Species

    H = Species("H", subdomains=[vol1])
    H2 = Species("H2", subdomains=[vol2])

    recombination = InterfaceReaction(
        id=3,
        subdomains=[vol1, vol2],
        k_plus=2,
        k_minus=1,
        reactants=[H, H],
        products=[H2],
    )

Competing reactions
-------------------

Several :class:`festim.InterfaceReaction` can share the same ``id``: their contributions are summed.
This models an interface with several exchange pathways.
For example, hydrogen leaving a metal into a molten fluoride salt can recombine into :math:`\mathrm{H}_2` or be oxidised into :math:`\mathrm{HF}`:

.. testcode:: interfaces

    from festim import InterfaceReaction, Species

    H = Species("H", subdomains=[vol1])
    H2 = Species("H2", subdomains=[vol2])
    HF = Species("HF", subdomains=[vol2])

    # 2 H (metal) <=> H2 (salt)
    recombination = InterfaceReaction(
        id=3, subdomains=[vol1, vol2], k_plus=2, k_minus=1, reactants=[H, H], products=[H2]
    )

    # H (metal) <=> HF (salt)
    fluorination = InterfaceReaction(
        id=3, subdomains=[vol1, vol2], k_plus=1, k_minus=1, reactants=[H], products=[HF]
    )

    interfaces = [recombination, fluorination]

Several isotopes are handled in the same way.
With H and T in the metal, the reactions :math:`2 \mathrm{H} \rightleftharpoons \mathrm{H}_2`, :math:`\mathrm{H} + \mathrm{T} \rightleftharpoons \mathrm{HT}` and :math:`2 \mathrm{T} \rightleftharpoons \mathrm{T}_2` couple the two isotopes through the mixed molecule HT, which cannot be described by an LTE condition.

When can LTE be used?
---------------------

LTE is the fast-kinetics limit of a single reaction: when :math:`k_+ \to \infty` with a fixed ratio :math:`k_+/k_-`, the interface reaction gives the LTE condition.
Two dimensionless numbers indicate whether this limit is reached.

The Damköhler number compares the interfacial exchange to the bulk diffusion:

.. math::

    \mathrm{Da} = \frac{k_\mathrm{eff} L}{D}

where :math:`L` and :math:`D` are the thickness and the diffusivity of the adjacent subdomain, and :math:`k_\mathrm{eff}` is the exchange velocity of the reaction: :math:`k_+` for a first-order exchange and :math:`2 k_+ c_\mathrm{H}` for recombination.
For recombination, :math:`k_\mathrm{eff}` depends on the interfacial concentration, so the same interface can be limited by the kinetics at low concentration and at equilibrium at high concentration.
When :math:`\mathrm{Da} \gtrsim 100`, the reaction is at equilibrium and the LTE condition can be used.

The branching ratio compares competing reactions.
For the recombination and fluorination example above, it is the ratio of the hydrogen fluxes carried by each reaction, :math:`\mathcal{B} = R_\mathrm{HF} / (2 R_{\mathrm{H}_2})`.
When one reaction dominates (:math:`\mathcal{B}` at least two orders of magnitude away from 1), its LTE condition can be used.
Otherwise, no single algebraic condition describes the interface, however fast the kinetics.

.. note::

    An interface reaction couples the two bulk fields directly: no hydrogen is stored at the interface.
    To model hydrogen trapped at the interface, use a codimensional subdomain with its own species (see :ref:`Codimensional (manifold) Subdomains`).
