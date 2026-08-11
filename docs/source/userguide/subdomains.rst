=======================
Subdomains & Materials
=======================

Subdomains define different regions within the simulation domain, each assigned specific physical models or materials.

Subdomains are categorized as:
1. **Surface subdomains**: Regions on the outer boundaries of the simulation domain.
2. **Volume subdomains**: Regions inside the simulation domain.

Surface Subdomains
==================

Use the :class:`festim.SurfaceSubdomain` class to define surface subdomains.


.. testcode::

    from festim import SurfaceSubdomain

    my_surface = SurfaceSubdomain(id=1)

The `id` is a unique identifier for the surface subdomain. It corresponds to mesh tags assigned to the model or can be set during mesh creation using external tools.

.. note::

    If no mesh tags are provided, the surface subdomain ID defaults to 1 on all outer boundaries.

For 1D domains, use the :class:`festim.SurfaceSubdomain1D` class, which requires an additional `x` argument to specify the surface position.

.. testcode::

    from festim import SurfaceSubdomain1D

    my_1D_surface = SurfaceSubdomain1D(id=1, x=10)


Custom surface subdomains can be created by subclassing the :class:`festim.SurfaceSubdomain` class. In this case we can use a custom unit square mesh, and would like to have a defined surface on the top of the domain where y=1.

.. testcode::

    from dolfinx.mesh import create_unit_square
    from mpi4py import MPI
    from festim import SurfaceSubdomain

    my_mesh = create_unit_square(MPI.COMM_WORLD, 50, 50)

    class TopSurface(SurfaceSubdomain):
        
        # Surface subdomains need a method to locate the facets of the mesh
        def locate_boundary_facet_indices(self, mesh):
            surface_dim = mesh.topology.dim - 1 # dimension of the surface of the domain

            # locate the facets of the mesh where y = 1 
            indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[1], 1)) 
            return indices

.. note::

    The different coordinates x, y, z are represented by x[0], x[1], x[2] in fenics, respectively.


Volume Subdomains
=================

Volume subdomains define distinct regions within the simulation domain and assign materials to these regions.

.. testcode::

    from festim import VolumeSubdomain, Material

    my_material = Material(D_0=1, E_D=1)
    my_volume = VolumeSubdomain(id=1, material=my_material)

For 1D domains, use the :class:`festim.VolumeSubdomain1D` class, which requires a `borders` argument to specify the domain boundaries where the material is applied.

.. testcode::

    from festim import VolumeSubdomain1D, Material

    my_material = Material(D_0=1, E_D=1)
    my_1D_volume = VolumeSubdomain1D(id=1, material=my_material, borders=[0, 1])

Codimensional (manifold) Subdomains
====================================

A volume subdomain can be a *manifold* embedded in the mesh: a line in a 2D mesh, or a
surface in a 3D mesh. Such a subdomain carries its own transport equation, with
diffusion and advection **along** the manifold, and is coupled to the bulk by a flux.
This is the way to model transport in a grain boundary, along a crack, or in a thin
surface layer you do not want to resolve with cells.

Pass ``dim`` one less than the dimension of the mesh:

.. testcode::

    import numpy as np
    from festim import VolumeSubdomain, Material

    gamma = VolumeSubdomain(
        id=2,
        material=Material(D_0=1e-6, E_D=0.0),
        dim=1,  # a line inside a 2D mesh
        locator=lambda x: np.isclose(x[0], 0.0),
    )

A manifold subdomain is tagged in the **facet** meshtags rather than the cell meshtags,
so its ``id`` must be unique among the surface subdomains as well. It can be used
directly wherever a surface is expected, for instance as the ``subdomain`` of a
:class:`festim.ParticleFluxBC` — there is no need to declare a separate
:class:`festim.SurfaceSubdomain` on the same facets.

A manifold may sit on the outer boundary of the domain, or *between* two volume
subdomains — a grain boundary, or an interface layer with its own trapping — in which
case it exchanges with both sides.

Coupling to the bulk
--------------------

The exchange is written twice: once as a flux leaving the bulk, and once as a source
entering the manifold. Use ``species_dependent_value`` to let each half see both
concentrations, even though they live on different meshes::

    k = 0.1
    J = lambda c_bulk, c_man: k * (c_bulk - c_man)

    # the bulk loses J through gamma
    flux = F.ParticleFluxBC(
        subdomain=gamma,
        value=lambda c_man, c_bulk: -J(c_bulk, c_man),
        species=H_bulk,
        species_dependent_value={"c_bulk": H_bulk, "c_man": H_manifold},
    )

    # ... and the manifold gains it
    source = F.ParticleSource(
        volume=gamma,
        value=lambda c_man, c_bulk: J(c_bulk, c_man),
        species=H_manifold,
        species_dependent_value={"c_bulk": H_bulk, "c_man": H_manifold},
    )

.. warning::

    **Mind the units.** A :class:`festim.ParticleFluxBC` value is a *flux* (H/m²/s for a
    3D bulk) whereas a :class:`festim.ParticleSource` value is a *volumetric rate*
    (H/m³/s). Writing the same expression on both sides is therefore not generally
    dimensionally consistent, and the manifold-side source usually needs a conversion
    factor.

    FESTIM does not impose a unit convention on a manifold species: depending on what
    you are modelling it may be a volumetric concentration H/m³ (a layer of thickness
    :math:`\lambda`, in which case the source is :math:`J/\lambda`), an areal density
    H/m² (an adsorbed layer, source :math:`J`), or a line density H/m (a grain
    boundary). Keeping the problem dimensionally consistent is up to you.

Manifolds between two subdomains
--------------------------------

When a manifold separates two volume subdomains, declare **one exchange per side** —
one :class:`festim.ParticleFluxBC` and one :class:`festim.ParticleSource` each. Both
name the same manifold as their subdomain; FESTIM works out which side each belongs to
from the bulk species it reads, so nothing else has to be specified::

    for bulk_species, k in ((H_left, k_left), (H_right, k_right)):
        J = lambda c_man, c_bulk: k * (c_bulk - c_man)
        bcs.append(F.ParticleFluxBC(
            subdomain=gamma, species=bulk_species,
            value=lambda c_man, c_bulk: -J(c_man, c_bulk),
            species_dependent_value={"c_bulk": bulk_species, "c_man": H_manifold}))
        sources.append(F.ParticleSource(
            volume=gamma, species=H_manifold, value=J,
            species_dependent_value={"c_bulk": bulk_species, "c_man": H_manifold}))

A single source may not read the bulk concentrations of *both* sides at once: an
interior manifold is integrated over interior facets, where each term has to be
restricted to one side. Split such a source in two, as above.

.. note::

    A pair of volume subdomains may be separated either by a
    :class:`festim.Interface` -- imposing a jump in concentration across a shared
    boundary -- or by a codim-1 subdomain carrying its own transport equation, but not
    both. FESTIM raises if an interface and a manifold cover the same facets.

Advection along a manifold
--------------------------

An :class:`festim.AdvectionTerm` on a manifold subdomain takes an ordinary ambient
velocity vector — 2 components in a 2D mesh, 3 in a 3D mesh. There is no need to
project it onto the manifold: the tangential gradient is orthogonal to the normal, so
:math:`v \cdot \nabla_\Gamma c` automatically ignores the normal component of
:math:`v`.

Boundary conditions on a manifold
---------------------------------

A manifold has a boundary of its own — the endpoints of a line in a 2D mesh, the rim of
a surface in a 3D mesh — and boundary conditions can be applied there. Declare it as a
:class:`festim.SurfaceSubdomain` with ``dim`` set to the mesh dimension minus **two**,
just as a manifold is a :class:`festim.VolumeSubdomain` with ``dim`` set to the mesh
dimension minus one:

.. code-block:: python

    # a 1D fluid running along a 2D pipe wall
    fluid = F.VolumeSubdomain(id=2, material=..., dim=1,
                              locator=lambda x: np.isclose(x[1], H))

    # the inlet: one end of that 1D domain
    inlet = F.SurfaceSubdomain(id=3, dim=0,
                               locator=lambda x: np.isclose(x[0], 0.0))

    ...
    boundary_conditions=[
        F.FixedConcentrationBC(subdomain=inlet, value=c_in, species=c_fluid),
    ]

The locator is evaluated on the manifold, not on the parent mesh, and must select a
point on its boundary — a locator matching only interior points raises rather than
silently doing nothing.

Such a surface carries no meshtag, so its ``id`` does not have to differ from a manifold
or interface id. Which manifold it bounds is taken from the ``species`` of the boundary
condition using it, so that species must live on exactly one manifold; the same surface
object can be reused on several manifolds, one species each.

Without a condition of this kind, the ends of a manifold carry the natural zero-flux
condition.

Reactions and trapping on a manifold
------------------------------------

A :class:`festim.Reaction` runs on a manifold like on any other volume subdomain: give
it ``volume=gamma`` and species that live on ``gamma``. Trapping is written as a
reaction against :class:`festim.ImplicitSpecies` empty sites:

.. code-block:: python

    trapped = F.Species("trapped", mobile=False, subdomains=[gamma])
    empty_sites = F.ImplicitSpecies(n=n_trap, others=[trapped])

    trapping = F.Reaction(
        reactant=[H_manifold, empty_sites], product=trapped,
        k_0=k_0, E_k=E_k, p_0=p_0, E_p=E_p, volume=gamma,
    )

Note that the trapped species is declared with its own ``subdomains``, and that
:class:`festim.Trap` is not a shortcut for this — it builds a species without one.

The density ``n`` of an implicit species consumed on a manifold is a coefficient of an
integral over that manifold, so FESTIM builds it there. Two consequences: give ``n`` as
a float or as a callable of ``x`` and ``t`` rather than as a ready-made
``dolfinx.fem.Function``, which cannot be moved; and declare one implicit species per
subdomain rather than sharing one between a reaction on a manifold and a reaction
elsewhere. Both are raised rather than silently mis-assembled.

Exports on a manifold
---------------------

Derived quantities can be asked for in three places once a manifold is in the mesh.

**Over the manifold**, for its own species — a volume quantity, with ``volume`` set to
the manifold. It is integrated over the manifold itself, so a
:class:`festim.TotalVolume` on a line in a 2D mesh is a line integral:

.. code-block:: python

    F.TotalVolume(field=c_gamma, volume=gamma)
    F.AverageVolume(field=c_gamma, volume=gamma)

**On the facets the manifold occupies**, for a *bulk* species — a surface quantity, with
the manifold passed where a surface subdomain normally goes. This is the exchange
between the bulk and the manifold:

.. code-block:: python

    F.SurfaceFlux(field=c_bulk, surface=gamma)
    F.TotalSurface(field=c_bulk, surface=gamma)

On an interior manifold, which side the quantity is read on follows from ``field``,
exactly as it does for the flux boundary conditions above — declare one export per side,
each naming that side's species. The sign convention is the ordinary one: a positive
flux leaves the subdomain the species lives on, so an exchange from the left bulk to the
right one through the manifold reads positive on the left and negative on the right.

Asking for a manifold's *own* species on its own facets raises: it has no flux across
the manifold, and the quantity meant is the volume one above.

**On the boundary of the manifold** — a surface quantity whose ``surface`` is the
codim-2 :class:`festim.SurfaceSubdomain` described in the previous section. This is what
gives the outlet flux of the pipe example:

.. code-block:: python

    outlet = F.SurfaceSubdomain(id=4, dim=0,
                                locator=lambda x: np.isclose(x[0], L))
    ...
    exports=[F.SurfaceFlux(field=c_fluid, surface=outlet)]

.. note::

    :class:`festim.SurfaceFlux` computes the **diffusive** flux only. On a manifold
    carrying an :class:`festim.AdvectionTerm` the advective part is not included, and
    FESTIM warns.

Limitations
-----------

* Only codimension 1 is supported (``dim`` must be the mesh dimension minus one), and a
  manifold must be adjacent to one volume subdomain (on the boundary of the domain) or
  two (on an interface). A codimension-2 subdomain carrying its own equation is not
  supported: a bulk field has no well-defined trace on a line in 3D or a point in 2D,
  so the exchange with it would not be well posed.
* Boundary conditions on the boundary of a manifold are limited to
  :class:`festim.FixedConcentrationBC`.
* Exports on and around a manifold are limited to the integral-based derived quantities
  (:class:`festim.SurfaceFlux`, :class:`festim.TotalSurface`,
  :class:`festim.AverageSurface`, :class:`festim.TotalVolume`,
  :class:`festim.AverageVolume`) and :class:`festim.VTXSpeciesExport`. The minimum and
  maximum quantities are not available in
  :class:`festim.HydrogenTransportProblemDiscontinuous` at all, manifold or not.
* Cartesian coordinates only.

----------
Materials
----------

Materials play a key role in hydrogen transport simulations, defining diffusivity, solubility, and thermal properties such as thermal conductivity and heat capacity.

To define a material, use the :class:`festim.Material` class:

.. testcode::

    from festim import Material

    mat = Material(D_0=2, E_D=0.1)

The :class:`festim.Material` class requires two arguments:

* :code:`D_0`: The diffusivity pre-exponential factor (m²/s).
* :code:`E_D`: The diffusivity activation energy (eV).

Diffusivity is automatically computed using these parameters based on the Arrhenius law.

Additional parameters are required for specific simulations. When considering chemical potential conservation at material interfaces, hydrogen solubility must be specified using:

* :code:`name`: Name for the material.
* :code:`S_0`: The solubility pre-exponential factor (units depend on the solubility law: Sievert's or Henry's).
* :code:`E_S`: The solubility activation energy (eV).
* :code:`solubility_law`: The solubility law, either :code:`"festim.SolubilityLaw.HENRY"` or :code:`festim.SolubilityLaw.SIEVERT`.

For transient heat transfer simulations, thermal conductivity, heat capacity, and density must be defined:

* :code:`thermal_conductivity`: Thermal conductivity (W/m/K).
* :code:`heat_capacity`: Heat capacity (J/kg/K).
* :code:`density`: Density (kg/m³).

Temperature-dependent Parameters
---------------------------------

Thermal properties can be defined as functions of temperature. For example:

.. testcode::

    from festim import Material
    import ufl

    my_mat = Material(
        name="my_fancy_material",
        D_0=2e-7,
        E_D=0.2,
        thermal_conductivity=lambda T: 3 * T + 2 * ufl.exp(-20 * T),
        heat_capacity=lambda T: 4 * T + 8,
        density=lambda T: 7 * T + 5,
    )

Integration with HTM
---------------------

H-transport-materials (HTM) is a Python database of hydrogen transport properties. Using HTM helps prevent copy-paste errors and ensures consistency across simulations by using standardised property values.

HTM can be easily `integrated with FESTIM <https://festim-workshop.readthedocs.io/en/latest/content/material/material_htm.html>`_.

.. note::

    This example demonstrates HTM integration with FESTIM v1.4, but the same principle applies to other versions.
