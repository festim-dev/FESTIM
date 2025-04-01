==========
Subdomains
==========

.. warning::

    🔨 This page is under construction. 🔨


Subdomains define different regions within the simulation domain, each assigned specific physical models or materials.

Subdomains are categorized as:
1. **Surface subdomains**: Regions on the outer boundaries of the simulation domain.
2. **Volume subdomains**: Regions inside the simulation domain.

------------------
Surface Subdomains
------------------

Use the :class:`festim.SurfaceSubdomain` class to define surface subdomains.
.. testsetup::

    from festim import SurfaceSubdomain

.. testcode::

    my_surface = SurfaceSubdomain(id=1)

The `id` is a unique identifier for the surface subdomain. It corresponds to mesh tags assigned to the model or can be set during mesh creation using external tools.

.. note::

    If no mesh tags are provided, the surface subdomain ID defaults to 1 on all outer boundaries.

For 1D domains, use the :class:`festim.SurfaceSubdomain1D` class, which requires an additional `x` argument to specify the surface position.

.. testsetup::

    from festim import SurfaceSubdomain1D

.. testcode::

    my_surface = SurfaceSubdomain1D(id=1, x=10)
    
------------------
Volume Subdomains
------------------

Volume subdomains define distinct regions within the simulation domain and assign materials to these regions.

.. testsetup::

    from festim import VolumeSubdomain, Material

.. testcode::

    my_material = Material(D_0=1, E_D=1)
    my_volume = VolumeSubdomain(id=1, material=my_material)

For 1D domains, use the :class:`festim.VolumeSubdomain1D` class, which requires a `borders` argument to specify the domain boundaries where the material is applied.

.. testsetup::

    from festim import VolumeSubdomain1D, Material

.. testcode::

    my_material = Material(D_0=1, E_D=1)
    my_volume = VolumeSubdomain1D(id=1, material=my_material, borders=[0, 1])

----------
Materials
----------

Materials play a key role in hydrogen transport simulations, defining diffusivity, solubility, and thermal properties such as thermal conductivity and heat capacity.

To define a material, use the :class:`festim.Material` class:

.. testsetup::

    from festim import Material

.. testcode::

    mat = Material(D_0=2, E_D=0.1)

The :class:`festim.Material` class requires two arguments:

* :code:`D_0`: The diffusivity pre-exponential factor (m²/s).
* :code:`E_D`: The diffusivity activation energy (eV).

Diffusivity is automatically computed using these parameters based on the Arrhenius law.

Additional parameters are required for specific simulations. When considering chemical potential conservation at material interfaces, hydrogen solubility must be specified using:

* :code:`name`: Name for the material.
* :code:`S_0`: The solubility pre-exponential factor (units depend on the solubility law: Sievert's or Henry's).
* :code:`E_S`: The solubility activation energy (eV).
* :code:`solubility_law`: The solubility law, either :code:`"henry"` or :code:`"sievert"`.

For transient heat transfer simulations, thermal conductivity, heat capacity, and density must be defined:

* :code:`thermal_conductivity`: Thermal conductivity (W/m/K).
* :code:`heat_capacity`: Heat capacity (J/kg/K).
* :code:`density`: Density (kg/m³).

---------------------------------
Temperature-dependent Parameters
---------------------------------

Thermal properties can be defined as functions of temperature. For example:

.. testsetup::

    from festim import Material
    import ufl

.. testcode::

    my_mat = Material(
        name="my_fancy_material",
        D_0=2e-7,
        E_D=0.2,
        thermal_conductivity=lambda T: 3 * T + 2 * ufl.exp(-20 * T),
        heat_capacity=lambda T: 4 * T + 8,
        density=lambda T: 7 * T + 5,
    )

--------------------
Integration with HTM
--------------------

H-transport-materials (HTM) is a Python database of hydrogen transport properties. Using HTM helps prevent copy-paste errors and ensures consistency across simulations by using standardised property values.

HTM can be easily `integrated with FESTIM <https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task08.ipynb>`_.

.. note::

    This example demonstrates HTM integration with FESTIM v1.4, but the same principle applies to other versions.