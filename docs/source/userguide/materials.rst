.. _materials_guide:

=========
Materials
=========

Materials are vital components of hydrogen transport simulations. They hold diffusivity, solubility and thermal properties like thermal conductivity or heat capacity.

To define a material, use the :class:`festim.Material` class:

.. testsetup::

    from festim import Material, Simulation
    import fenics

.. testcode::

    mat = Material(id=1, D_0=2, E_D=0.1)

The :class:`festim.Material` class has three required arguments:

* :code:`id`: a unique id given to the material/volume. It is useful when defining volumetric source terms or exports. Several id's can be given to the same material if multiple volumes have the same material.
* :code:`D_0`: the diffusivity pre-exponential factor expressed in m2/s
* :code:`E_D`: the diffusivity activation energy in eV

The diffusivity will be automatically evaluated using the pre-exponential factor and activation energy according to the Arrhenius law.

The material is then assigned to the model:

.. testcode::

    my_model = Simulation(materials=mat)

Similarly, several materials can be used in simulations:

.. testcode::

    mat1 = Material(id=1, D_0=2, E_D=0.1)
    mat2 = Material(id=2, D_0=3, E_D=0.4)
    my_model = Simulation(materials=[mat1, mat2])

.. note::

    When several materials are considered in one-dimensional simulations, the ``borders`` argument needs to be provided for each material:

    .. testcode::

        mat1 = Material(id=1, D_0=2, E_D=0.1, borders=[0, 0.5])
        mat2 = Material(id=2, D_0=3, E_D=0.4, borders=[0.5, 1.0])
    
    ``borders`` determine the domain where the material is defined.
    

Some other parameters are optional and are only required for specific types of simulations. The hydrogen solubility in a material needs to be provided 
when the conservation of chemical potential at interfaces of materials is considered. It is defined by the following parameters:

* :code:`S_0`: the solubility pre-exponential factor, its units depend on the solubility law (Sievert's or Henry)
* :code:`E_S`: the solubility activation energy in eV
* :code:`solubility_law`: the material’s solubility law. Can be :code:`“henry”` or :code:`“sievert”`

For transient heat transfer simulations, thermal conductivity, heat capacity, and density of a material are required. They can be set using the corresponding  
material attributes:

* :code:`thermal_cond`: the thermal conductivity in W/m/K
* :code:`heat_capacity`: the heat capacity in J/kg/K
* :code:`rho`: the volumetric density in kg/m3

Finally, the :ref:`Soret effect` can be accounted for by invoking:

* :code:`Q`: the heat of transport in eV.

---------------------------------
Temperature-dependent parameters
---------------------------------

Thermal properties and the heat of transport can be defined as function of temperature. For example:

.. testcode::

    my_mat = Material(
        id=1,
        D_0=2e-7,
        E_D=0.2,
        thermal_cond=lambda T: 3 * T + 2 * fenics.exp(-20 * T),
        heat_capacity=lambda T: 4 * T + 8 * fenics.conditional(T < 400, 5, 8),
        rho=lambda T: 7 * T + 5,
        Q=lambda T: -0.5 * T**2,
    )

--------------------
Integration with HTM
--------------------

H-transport-materials (HTM) is a Python database of hydrogen transport properties.
Using this database will avoid making copy-pasting errors and add consistency across simulations by making sure the same properties are used.
HTM can be easily `integrated with FESTIM <https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task08.ipynb>`_.
