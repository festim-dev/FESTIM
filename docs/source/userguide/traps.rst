.. _traps:

=====
Traps
=====

------------
Basic traps
------------


A trap in FESTIM is defined by:

* its trapping coefficient
* its detrapping coefficient
* its density
* the materials where it is located

.. testcode::

    import festim as F

    my_material = F.Material(id=1, D_0=1, E_D=0)

    my_trap = F.Trap(k_0=1e-16, E_k=0.2, p_0=1e13, E_p=0.8, density=1e16, materials=my_material)

If the trap is located in several materials, instead of creating another :class:`festim.Trap` object, simply use a list of materials:

.. testcode::

    import festim as F

    mat1 = F.Material(id=1, D_0=1, E_D=0)
    mat2 = F.Material(id=2, D_0=2, E_D=0)

    my_trap = F.Trap(k_0=1e-16, E_k=0.2, p_0=1e13, E_p=0.8, density=1e16, materials=[mat1, mat2])

The trap density can be a function of space and time. For example:

.. testcode::

    import festim as F

    my_trap = F.Trap(
        k_0=1e-16,
        E_k=0.2,
        p_0=1e13,
        E_p=0.8,
        density=1e16 + F.t + F.x,
        materials=my_material
    )

Boolean expressions can also be used to restrict the trap to certain regions:

.. testcode::

    import festim as F

    import sympy as sp

    my_trap = F.Trap(
        k_0=1e-16,
        E_k=0.2,
        p_0=1e13,
        E_p=0.8,
        density=sp.Piecewise((1e16, F.x < 0.1), (0, True)),
        materials=my_material
    )

In this case, the trap's density will be :math:`10^{16} \ \mathrm{m^{-3}}` for all :math:`x < 0.1 \ \mathrm{m}`, else zero.

---------------
Extrinsic traps
---------------

An extrinsic trap is defined as a trap with a density evolving over time.
If the temporal evolution of the trap's density is known `a priori`, then a "normal" trap can be used with a time dependent expression as density (see above).

.. testcode::

    import festim as F

    mat1 = F.Material(id=1, D_0=1, E_D=0)
    mat2 = F.Material(id=2, D_0=2, E_D=0)

    trap1 = F.Trap(k_0=1e-16, E_k=0.2, p_0=1e13, E_p=0.8, density=1e16, materials=mat1)
    trap2 = F.Trap(k_0=1e-16, E_k=0.2, p_0=1e13, E_p=1.0, density=1e16, materials=mat2)

------------
Grouped-trap
------------

Let's imagine a case where you have two subdomains. Trap 1 is defined only in the first subdomain, whereas Trap 2 is defined in the second.
It would be possible to define one trap in each subdomain.
Grouping traps together helps save computational time by reducing the number of degrees of freedom.

.. testcode::

    import festim as F

    mat1 = F.Material(id=1, D_0=1, E_D=0)
    mat2 = F.Material(id=2, D_0=2, E_D=0)

    grouped_trap = F.Trap(
        k_0=[1e-16, 1e-16],
        E_k=[0.2, 0.2],
        p_0=[1e13, 1e13],
        E_p=[0.8, 1.0],
        density=[1e16, 1e16],
        materials=[mat1, mat2],
    )

