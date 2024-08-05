.. _boundary_conditions_guide:

===================
Boundary conditions
===================

.. testsetup:: BCs

    from festim import *

The boundary conditions (BCs) are essential to FESTIM simulations. They describe the mathematical problem at the boundaries of the simulated domain.
If no BC is set on a boundary, it is assumed that the flux is null. This is also called a symmetry BC.

---------------
Basic BCs
---------------
These BCs can be used for heat transfer or hydrogen transport simulations.

Imposing the solution
^^^^^^^^^^^^^^^^^^^^^

The value of solutions (concentration, temperature) can be imposed on boundaries with :class:`festim.DirichletBC`: 

.. testcode:: BCs

    my_bc = DirichletBC(surfaces=[2, 4], value=10, field=0)

.. note::

    Here, we set :code:`field=0` to specify that this BC applies to the mobile hydrogen concentration. :code:`1` would stand for the trap 1 concentration, and :code:`"T"` for temperature.

The `value` argument can be space and time dependent by making use of the FESTIM variables ``x``, ``y``, ``z`` and ``t``:

.. testcode:: BCs

    from festim import x, y, z, t
    my_bc = DirichletBC(surfaces=3, value=10 + x**2 + t, field="T")


To use more complicated mathematical expressions, you can use the sympy package:

.. testcode:: BCs

    from festim import x, y, z, t
    import sympy as sp

    my_bc = DirichletBC(surfaces=3, value=10*sp.exp(-t), field="T")

- CustomDirichlet

The value of the concentration field can be temperature-dependent (useful when dealing with heat-transfer solvers) with :class:`festim.CustomDirichlet`:

.. testcode:: BCs

    def value(T):
        return 3*T + 2

    my_bc = CustomDirichlet(surfaces=3, function=value, field=0)

Imposing the flux
^^^^^^^^^^^^^^^^^

When the flux needs to be imposed on a boundary, use the :class:`festim.FluxBC` class.


.. testcode:: BCs

    my_bc = FluxBC(surfaces=3, value=10 + x**2 + t, field="T")


As for the Dirichlet boundary conditions, the flux can be dependent on temperature and mobile hydrogen concentration:

.. testcode:: BCs

    def value(T, mobile):
        return mobile**2 + T

    my_bc = CustomFlux(surfaces=3, function=value, field=0)


----------------------
Hydrogen transport BCs
----------------------

Some BCs are specific to hydrogen transport. FESTIM provides a handful of convenience classes making things a bit easier for the users.

Recombination flux
^^^^^^^^^^^^^^^^^^

A recombination flux can be set on boundaries as follows: :math:`Kr \, c_\mathrm{m}^n` (See :class:`festim.RecombinationFlux`).
Where :math:`Kr` is the recombination coefficient, :math:`c_\mathrm{m}` is the mobile hydrogen concentration and :math:`n` is the recombination order.

.. testcode:: BCs

    my_bc = RecombinationFlux(surfaces=3, Kr_0=2, E_Kr=0.1, order=2)


Dissociation flux
^^^^^^^^^^^^^^^^^^

Dissociation flux can be set on boundaries as: :math:`Kd \, P` (see :class:`festim.DissociationFlux`).
Where :math:`Kd` is the dissociation coefficient, :math:`P` is the partial pressure of hydrogen.

.. testcode:: BCs

    my_bc = DissociationFlux(surfaces=2, Kd_0=2, E_Kd=0.1, P=1e05)

Kinetic surface model (1D)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Kinetic surface model can be included to account for the evolution of adsorbed hydrogen on a surface with the :class:`festim.SurfaceKinetics` class.
The current class is supported for 1D simulations only. Refer to the :ref:`Kinetic surface model` theory section for more details.

.. testcode:: BCs

    from festim import t
    import fenics as f

    def k_bs(T, surf_conc, t):
        return 1e13*f.exp(-0.2/k_b/T)

    def k_sb(T, surf_conc, t):
        return 1e13*f.exp(-1.0/k_b/T)

    def J_vs(T, surf_conc, t):

        J_des = 2e5*surf_conc**2*f.exp(-1.2/k_b/T)
        J_ads = 1e17*(1-surf_conc/1e17)**2*f.conditional(t<10, 1, 0)

        return J_ads - J_des

    my_bc = SurfaceKinetics(
        k_bs=k_bs,
        k_sb=k_sb,
        lambda_IS=1.1e-10,
        n_surf=1e17,
        n_IS=6.3e28,
        J_vs=J_vs,
        surfaces=3,
        initial_condition=0,
        t=t
        )

Sievert's law of solubility
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Impose the mobile concentration of hydrogen as :math:`c_\mathrm{m} = S(T) \sqrt{P}` where :math:`S` is the Sievert's solubility and :math:`P` is the partial pressure of hydrogen (see :class:`festim.SievertsBC`).

.. testcode:: BCs

    from festim import t

    my_bc = SievertsBC(surfaces=3, S_0=2, E_S=0.1, pressure=2 + t)


Henry's law of solubility
^^^^^^^^^^^^^^^^^^^^^^^^^

Similarly, the mobile concentration can be set from Henry's law of solubility :math:`c_\mathrm{m} = K_H P` where :math:`K_H` is the Henry solubility (see :class:`festim.HenrysBC`).


.. testcode:: BCs

    from festim import t

    my_bc = HenrysBC(surfaces=3, H_0=2, E_H=0.1, pressure=2 + t)

Plasma implantation approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plasma implantation can be approximated by a Dirichlet boundary condition with the class :class:`festim.ImplantationDirichlet` . Refer to the :ref:`theory` section for more details.


.. testcode:: BCs

    from festim import t

    # instantaneous recombination
    my_bc = ImplantationDirichlet(surfaces=3, phi=1e10 + t, R_p=1e-9, D_0=1, E_D=0.1)

    # non-instantaneous recombination
    my_bc = ImplantationDirichlet(surfaces=3, phi=1e10 + t, R_p=1e-9, D_0=1, E_D=0.1, Kr_0=2, E_Kr=0.2)

    # non-instantaneous recombination and dissociation
    my_bc = ImplantationDirichlet(surfaces=3, phi=1e10 + t, R_p=1e-9, D_0=1, E_D=0.1, Kr_0=2, E_Kr=0.2, Kd_0=3, E_Kd=0.3, P=4)

-----------------
Heat transfer BCs
-----------------


A convective heat flux can be set as :math:`\mathrm{flux} = - h (T - T_\mathrm{ext})` (see :class:`festim.ConvectiveFlux`).

.. testcode:: BCs

    from festim import t

    my_bc = ConvectiveFlux(surfaces=3, h_coeff=0.1, T_ext=600 + 10*t)
